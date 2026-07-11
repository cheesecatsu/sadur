"""
Cek apakah embedding yang tersimpan di DB untuk satu baris tertentu
benar-benar cocok dengan teksnya (atau ke-swap sama baris lain).

Cara pakai:
    python cek_embedding_match.py
"""

import json
import numpy as np
import mysql.connector
from sentence_transformers import SentenceTransformer


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


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print("Memuat model bge-m3...")
    embedder = SentenceTransformer("BAAI/bge-m3")

    db = get_db()
    curr = db.cursor()
    curr.execute(
        "SELECT text, embedding FROM DOCS WHERE text LIKE %s",
        ("%karangan%",),
    )
    rows = curr.fetchall()
    curr.close()

    if not rows:
        print("⚠️  Gak ada baris yang mengandung kata 'karangan' sama sekali.")
        print("   Berarti chunk itu kemungkinan besar memang tidak pernah masuk ke DOCS.")
        return

    print(f"Ditemukan {len(rows)} baris yang mengandung 'karangan':\n")
    for i, (text, _) in enumerate(rows, 1):
        print(f"{i}. {text[:200]}...\n")

    # Pilih baris pertama yang juga mengandung 'esai' untuk dites lebih lanjut
    candidates = [r for r in rows if "esai" in r[0].lower()]
    if not candidates:
        print("Gak ada yang juga mengandung kata 'esai'. Cek manual daftar di atas.")
        return

    stored_text, stored_embedding_raw = candidates[0]
    print(f"\n--- Menguji kandidat pertama ---")

    # embedding di TiDB biasanya balik sebagai string JSON atau bytes, coba parse
    if isinstance(stored_embedding_raw, (bytes, bytearray)):
        stored_embedding_raw = stored_embedding_raw.decode("utf-8")
    stored_embedding = json.loads(stored_embedding_raw) if isinstance(stored_embedding_raw, str) else stored_embedding_raw

    print(f"\nTeks tersimpan: {stored_text[:150]}...")
    print(f"Panjang vektor tersimpan: {len(stored_embedding)}")

    fresh_embedding = embedder.encode(stored_text).tolist()
    print(f"Panjang vektor baru: {len(fresh_embedding)}")

    sim = cosine_sim(stored_embedding, fresh_embedding)
    print(f"\nCosine similarity (stored vs fresh dari teks yang SAMA): {sim:.4f}")

    if sim > 0.95:
        print("-> Embedding tersimpan MEMANG cocok dengan teksnya. Berarti masalahnya bukan di sini.")
    else:
        print("-> Embedding tersimpan TIDAK cocok dengan teksnya (harusnya mendekati 1.0).")
        print("   Ini konfirmasi ada bug pairing text<->embedding di script ingestion kamu.")


if __name__ == "__main__":
    main()