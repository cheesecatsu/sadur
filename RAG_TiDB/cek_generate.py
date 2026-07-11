"""
Tes langsung fungsi jawab tanpa lewat UI Streamlit, biar keliatan
persis prompt yang dikirim ke Ollama dan apa balasannya.

Cara pakai:
    python cek_generate.py
"""

import json
import re
import ollama
import mysql.connector
from sentence_transformers import SentenceTransformer

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"


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


def extract_answer_only(text):
    match = re.search(r"Jawaban:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text


def search_document(query, embedder, k_top=5, max_distance=0.55):
    db = get_db()
    curr = db.cursor()
    query_embedding = embedder.encode(query).tolist()
    query_embedding_str = json.dumps(query_embedding)
    curr.execute(
        f"""
        SELECT text, vec_cosine_distance(embedding, %s) AS distance
        FROM DOCS
        ORDER BY distance ASC
        LIMIT {k_top}
        """,
        (query_embedding_str,),
    )
    results = curr.fetchall()
    curr.close()
    return [
        {"text": text, "distance": distance}
        for text, distance in results
        if distance <= max_distance
    ]


def main():
    query = "apa itu esai"

    print("Memuat model embedder...")
    embedder = SentenceTransformer("BAAI/bge-m3")

    print("Mencari dokumen...")
    docs = search_document(query, embedder)
    print(f"Ditemukan {len(docs)} dokumen relevan.\n")

    context = "\n\n".join(extract_answer_only(d["text"]) for d in docs)

    prompt = f"""Jawab pertanyaan berdasarkan informasi berikut. Kamu boleh merangkum atau menyimpulkan dari informasi yang ada, tidak harus tertulis persis.
Hanya bilang 'Maaf, informasi ini di luar cakupan materi yang saya miliki' kalau informasi di bawah ini benar-benar tidak berkaitan dengan pertanyaan.

INFORMASI:
{context}

PERTANYAAN: {query}

JAWABAN:"""

    print("=" * 60)
    print("PROMPT YANG DIKIRIM KE MODEL:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)

    llm = ollama.Client(host=OLLAMA_HOST)
    response = llm.generate(model=OLLAMA_MODEL, prompt=prompt, stream=False)

    print("\nJAWABAN MODEL:")
    print(response["response"])


if __name__ == "__main__":
    main()