#!/usr/bin/env python3
"""
Tambahkan kolom `source` ke tabel DOCS pada TiDB Cloud, lalu isi otomatis
berdasarkan kolom `source` pada knowledge_base_final.csv.

Pencocokan dilakukan dengan urutan:
1. Isi kolom DOCS.text dicocokkan dengan CSV.formatted_text setelah normalisasi.
2. Jika tidak cocok, bagian "Pertanyaan:" pada DOCS.text dicocokkan dengan CSV.question.

Skrip bersifat idempotent: aman dijalankan ulang tanpa membuat kolom ganda
atau mengubah vector embedding.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import certifi
import mysql.connector


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{label} tidak valid: {value!r}")
    return value


def normalize_text(value: Any) -> str:
    """Samakan Unicode, spasi, dan line ending agar pencocokan lebih stabil."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def extract_question(text: str) -> str:
    """Ambil bagian di antara 'Pertanyaan:' dan 'Jawaban:'."""
    match = re.search(
        r"Pertanyaan:\s*(.*?)\s*Jawaban:",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return normalize_text(match.group(1)) if match else ""


def load_csv_maps(csv_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    required_columns = {"question", "formatted_text", "source"}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        missing = required_columns - fieldnames
        if missing:
            raise ValueError(
                "Kolom CSV tidak lengkap. Kolom yang belum ada: "
                + ", ".join(sorted(missing))
            )

        formatted_map: dict[str, str] = {}
        question_map: dict[str, str] = {}

        for row_number, row in enumerate(reader, start=2):
            formatted_text = normalize_text(row.get("formatted_text"))
            question = normalize_text(row.get("question"))
            source = str(row.get("source") or "").strip()

            if not formatted_text or not question or not source:
                print(
                    f"[PERINGATAN] Baris {row_number} dilewati karena "
                    "formatted_text, question, atau source kosong."
                )
                continue

            existing = formatted_map.get(formatted_text)
            if existing and existing != source:
                raise ValueError(
                    f"Konflik source untuk formatted_text pada baris {row_number}."
                )

            existing_question = question_map.get(question)
            if existing_question and existing_question != source:
                raise ValueError(
                    f"Konflik source untuk question pada baris {row_number}."
                )

            formatted_map[formatted_text] = source
            question_map[question] = source

    if not formatted_map:
        raise ValueError("Tidak ada data valid yang dapat dipakai dari CSV.")

    return formatted_map, question_map


def load_secrets_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Python < 3.11 memerlukan paket `tomli` untuk membaca secrets.toml."
            ) from exc

    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_db_config(secrets_path: Path) -> dict[str, Any]:
    """
    Prioritas konfigurasi:
    1. Environment variable
    2. .streamlit/secrets.toml
    """
    secrets = load_secrets_file(secrets_path)

    config = {
        "host": os.getenv("TIDB_HOST") or secrets.get("TIDB_HOST"),
        "port": int(os.getenv("TIDB_PORT") or secrets.get("TIDB_PORT", 4000)),
        "user": os.getenv("TIDB_USER") or secrets.get("TIDB_USER"),
        "password": os.getenv("TIDB_PASSWORD") or secrets.get("TIDB_PASSWORD"),
        "database": os.getenv("TIDB_DATABASE")
        or secrets.get("TIDB_DATABASE", "RAG"),
    }

    missing = [
        key for key in ("host", "user", "password", "database") if not config[key]
    ]
    if missing:
        raise RuntimeError(
            "Konfigurasi TiDB belum lengkap: "
            + ", ".join(missing)
            + ". Isi environment variable atau .streamlit/secrets.toml."
        )

    return config


def connect_db(config: dict[str, Any]):
    return mysql.connector.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        ssl_ca=certifi.where(),
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        autocommit=False,
    )


def ensure_source_column(cursor, table_name: str) -> bool:
    cursor.execute(f"SHOW COLUMNS FROM `{table_name}` LIKE 'source'")
    exists = cursor.fetchone() is not None

    if exists:
        print("[INFO] Kolom `source` sudah ada. Tidak perlu membuat ulang.")
        return False

    cursor.execute(f"ALTER TABLE `{table_name}` ADD COLUMN `source` TEXT NULL")
    print("[INFO] Kolom `source` berhasil ditambahkan.")
    return True


def migrate_sources(
    csv_path: Path,
    secrets_path: Path,
    table_name: str,
    id_column: str,
    text_column: str,
    unmatched_output: Path,
) -> None:
    table_name = validate_identifier(table_name, "Nama tabel")
    id_column = validate_identifier(id_column, "Nama kolom ID")
    text_column = validate_identifier(text_column, "Nama kolom teks")

    formatted_map, question_map = load_csv_maps(csv_path)
    config = load_db_config(secrets_path)

    connection = connect_db(config)
    cursor = connection.cursor()

    try:
        ensure_source_column(cursor, table_name)
        connection.commit()

        cursor.execute(
            f"SELECT `{id_column}`, `{text_column}`, `source` "
            f"FROM `{table_name}`"
        )
        rows = cursor.fetchall()

        updates: list[tuple[str, Any]] = []
        unmatched_rows: list[tuple[Any, str]] = []
        matched_exact = 0
        matched_question = 0
        unchanged = 0

        for docs_id, text, current_source in rows:
            normalized_db_text = normalize_text(text)
            source = formatted_map.get(normalized_db_text)

            if source:
                matched_exact += 1
            else:
                question = extract_question(normalized_db_text)
                source = question_map.get(question) if question else None
                if source:
                    matched_question += 1

            if not source:
                unmatched_rows.append((docs_id, str(text or "")))
                continue

            if str(current_source or "").strip() == source:
                unchanged += 1
                continue

            updates.append((source, docs_id))

        if updates:
            cursor.executemany(
                f"UPDATE `{table_name}` "
                f"SET `source` = %s WHERE `{id_column}` = %s",
                updates,
            )

        connection.commit()

        with unmatched_output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([id_column, text_column])
            writer.writerows(unmatched_rows)

        print("\n===== HASIL MIGRASI =====")
        print(f"Jumlah baris database       : {len(rows)}")
        print(f"Cocok berdasarkan teks      : {matched_exact}")
        print(f"Cocok berdasarkan pertanyaan: {matched_question}")
        print(f"Source diperbarui           : {len(updates)}")
        print(f"Sudah sesuai/tidak berubah  : {unchanged}")
        print(f"Tidak ditemukan di CSV      : {len(unmatched_rows)}")
        print(f"Laporan baris tidak cocok   : {unmatched_output.resolve()}")

        if unmatched_rows:
            print(
                "\n[CATATAN] Periksa file laporan. Baris yang tidak cocok "
                "tidak dihapus dan vector embedding tetap aman."
            )

    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tambahkan dan isi kolom source pada tabel DOCS TiDB."
    )
    parser.add_argument(
        "--csv",
        default="knowledge_base_final.csv",
        help="Lokasi knowledge base CSV.",
    )
    parser.add_argument(
        "--secrets",
        default=".streamlit/secrets.toml",
        help="Lokasi file Streamlit secrets.",
    )
    parser.add_argument("--table", default="DOCS", help="Nama tabel TiDB.")
    parser.add_argument(
        "--id-column", default="docs_id", help="Nama kolom primary key."
    )
    parser.add_argument(
        "--text-column", default="text", help="Nama kolom teks basis pengetahuan."
    )
    parser.add_argument(
        "--unmatched-output",
        default="unmatched_database_rows.csv",
        help="Lokasi laporan data database yang tidak cocok dengan CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        migrate_sources(
            csv_path=Path(args.csv),
            secrets_path=Path(args.secrets),
            table_name=args.table,
            id_column=args.id_column,
            text_column=args.text_column,
            unmatched_output=Path(args.unmatched_output),
        )
        return 0
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
