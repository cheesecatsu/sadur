import mysql.connector
import json
import ollama
from sentence_transformers import SentenceTransformer

# ===== KONFIGURASI =====
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

# ===== INISIALISASI =====
print("📥 Memuat model...")
llm_agent = ollama.Client(host=OLLAMA_HOST)
embedder = SentenceTransformer('BAAI/bge-m3')
print("✅ Model siap!\n")

# ===== KONEKSI DATABASE =====
db = mysql.connector.connect(
    host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    port=4000,
    user="4Qz9gBVCQx5NSYQ.root",
    password="IjuVu93yQQjJ65ND",
    database="RAG",
    ssl_ca="/etc/ssl/cert.pem",
    ssl_verify_cert=True,
    ssl_verify_identity=True
)
curr = db.cursor()

# ===== FUNGSI: SEARCH =====
def search_document(query, k_top=3):
    """Cari dokumen paling relevan"""
    query_embedding = embedder.encode(query).tolist()
    query_embedding_str = json.dumps(query_embedding)
    
    sql_query = f"""
        SELECT text, vec_cosine_distance(embedding, %s) AS distance
        FROM DOCS
        ORDER BY distance ASC
        LIMIT {k_top}
    """
    
    curr.execute(sql_query, (query_embedding_str,))
    results = curr.fetchall()
    
    # Ambil teks aja
    return [text for text, _ in results]

SYSTEM_PROMPT = """Kamu adalah asisten yang ramah dan membantu.
Gunakan bahasa Indonesia yang sopan tapi tidak kaku — seperti obrolan profesional yang hangat.
Hindari frasa template seperti "Berdasarkan informasi yang diberikan" atau "Sebagai asisten AI".
Kalau ada konteks yang diberikan, gunakan sebagai dasar jawaban tanpa menyebutnya secara eksplisit.
Kalau tidak tahu, katakan dengan jujur dan wajar.""" # Atur biar gak lolos ngecek yang lain

chat_history = [{'role': 'system', 'content': SYSTEM_PROMPT}]

# ===== FUNGSI: JAWAB =====
def jawab(query):
    docs = search_document(query, k_top=3)

    if docs:
        context = "\n\n".join(docs)
        user_message = f"Konteks:\n{context}\n\nPertanyaan: {query}"
    else:
        user_message = query

    chat_history.append({'role': 'user', 'content': user_message})

    response = llm_agent.chat(
        model=OLLAMA_MODEL,
        messages=chat_history
    )

    answer = response['message']['content']
    chat_history.append({'role': 'assistant', 'content': answer})

    return answer

# ===== MAIN =====
if __name__ == "__main__":
    curr.execute("SELECT COUNT(*) FROM DOCS")
    count = curr.fetchone()[0]
    print(f"Halo! Ada {count} dokumen tersedia. Silakan tanyakan apa saja.\n(ketik 'exit' untuk keluar)\n")
    
    while True:
        query = input("Prompt: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Sampai jumpa!")
            break

        try:
            jawaban = jawab(query)
            print(f"\n{jawaban}\n")

        except Exception as e:
            print(f"Aduh, ada error nih: {e}")

    curr.close()
    db.close()