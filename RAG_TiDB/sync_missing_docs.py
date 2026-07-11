"""
Bandingkan knowledge_base_final.csv dengan isi tabel DOCS di TiDB.
Baris yang ada di CSV tapi belum ada di DOCS akan di-embed dan di-insert.

Cara pakai:
    1. Taruh knowledge_base_final.csv di folder yang sama dengan script ini
       (atau ubah CSV_PATH di bawah).
    2. python sync_missing_docs.py           -> cuma cek & tampilkan yang hilang
    3. python sync_missing_docs.py --insert  -> cek DAN langsung insert yang hilang
"""

import sys
import json
import re
import pandas as pd
import mysql.connector
from sentence_transformers import SentenceTransformer

CSV_PATH = "knowledge_base_final.csv"


def get_db():
    return mysql.connector.connect(
        host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
        port=4000,
        user="4Qz9gBVCQx5NSYQ.root",
        password="IjuVu93yQQjJ65ND",
        database="RAG",
        ssl_ca="/etc/ssl/cert.pem",
        ssl_verify_cert=True,
        ssl_verify_identity=True,
    )


def normalize(text):
    """Buang semua whitespace & lowercase, biar perbandingan gak kena
    masalah spasi ganda/hilang seperti yang kejadian di sebagian data."""
    return re.sub(r"\s+", "", text).lower()


def main():
    do_insert = "--insert" in sys.argv

    print("Membaca CSV...")
    df = pd.read_csv(CSV_PATH)
    if "formatted_text" not in df.columns:
        print("⚠️  Kolom 'formatted_text' tidak ada di CSV. Cek nama kolom di file.")
        return
    print(f"Total baris di CSV: {len(df)}")

    print("\nMengambil semua teks yang sudah ada di DOCS...")
    db = get_db()
    curr = db.cursor()
    curr.execute("SELECT text FROM DOCS")
    existing_texts = [r[0] for r in curr.fetchall()]
    curr.close()
    print(f"Total baris di DOCS sekarang: {len(existing_texts)}")

    existing_normalized = [normalize(t) for t in existing_texts]

    missing_rows = []
    for _, row in df.iterrows():
        ft = row["formatted_text"]
        norm_ft = normalize(ft)
        found = any(norm_ft == e or norm_ft in e or e in norm_ft for e in existing_normalized)
        if not found:
            missing_rows.append(ft)

    print(f"\nJumlah baris CSV yang TIDAK ditemukan di DOCS: {len(missing_rows)}")
    for i, ft in enumerate(missing_rows, 1):
        print(f"{i}. {ft[:120]}...")

    if not missing_rows:
        print("\nGak ada yang hilang. Aman.")
        return

    if not do_insert:
        print("\nJalanin lagi dengan flag --insert kalau daftar di atas sudah benar,")
        print("supaya baris-baris ini di-embed dan dimasukkan ke DOCS.")
        return

    print("\nMemuat model bge-m3 untuk embedding...")
    embedder = SentenceTransformer("BAAI/bge-m3")

    db = get_db()
    curr = db.cursor()
    inserted = 0
    for ft in missing_rows:
        embedding = embedder.encode(ft).tolist()
        embedding_str = json.dumps(embedding)
        curr.execute(
            "INSERT INTO DOCS (text, embedding) VALUES (%s, %s)",
            (ft, embedding_str),
        )
        inserted += 1
    db.commit()
    curr.close()
    print(f"\n✅ Berhasil insert {inserted} baris yang sebelumnya hilang.")


if __name__ == "__main__":
    main()