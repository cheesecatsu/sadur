import mysql.connector
import json
from sentence_transformers import SentenceTransformer

# copy fungsi search_document + koneksi DB dari app.py kamu
embedder = SentenceTransformer('BAAI/bge-m3')

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

def search_document(query, k_top=5):
    curr = db.cursor()
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
    curr.close()
    return [{"text": text, "distance": distance} for text, distance in results]

docs = search_document("Bapak Ir. Soekarno di sebut sebagai Bapak proklamator, dia lahir pada tanggal 6 juni 1901.", k_top=5)
for d in docs:
    print(d["distance"], d["text"][:150])