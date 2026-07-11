"""
Script diagnostik: cek apa retrieval berhasil narik chunk yang relevan
untuk suatu pertanyaan, dan lihat skor jarak (distance) mentahnya.

Cara pakai:
    python cek_retrieval.py "apa itu esai"
"""

import sys
import json
import mysql.connector
from sentence_transformers import SentenceTransformer

OLLAMA_HOST = "http://localhost:11434"  # tidak dipakai di sini, cuma referensi

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

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "apa itu esai"
    print(f"Query: {query!r}\n")

    print("Memuat model bge-m3 (bisa agak lama pertama kali)...")
    embedder = SentenceTransformer("BAAI/bge-m3")

    query_embedding = embedder.encode(query).tolist()
    query_embedding_str = json.dumps(query_embedding)

    db = get_db()
    curr = db.cursor()

    # Ambil top-10, bukan top-3, biar keliatan apa chunk yang bener
    # muncul di ranking bawah atau tidak ada sama sekali di DB.
    curr.execute(
        """
        SELECT text, vec_cosine_distance(embedding, %s) AS distance
        FROM DOCS
        ORDER BY distance ASC
        LIMIT 10
        """,
        (query_embedding_str,),
    )
    results = curr.fetchall()
    curr.close()

    if not results:
        print("⚠️  Tabel DOCS kosong atau query gagal — cek koneksi/isi tabel.")
        return

    print(f"Ditemukan {len(results)} hasil teratas:\n")
    for i, (text, distance) in enumerate(results, 1):
        preview = text[:150].replace("\n", " ")
        print(f"{i}. distance={distance:.4f}  |  {preview}...")

    print("\n--- Cek manual ---")
    print("Cari baris yang isinya 'Esai adalah karangan yang mengemukakan pendapat...'")
    print("Kalau ADA di 10 hasil di atas -> retrieval sebenarnya jalan, tinggal atur k_top/threshold.")
    print("Kalau TIDAK ADA sama sekali    -> berarti chunk itu gagal diindex ke TiDB (masalah ingestion), bukan masalah query.")


if __name__ == "__main__":
    main()