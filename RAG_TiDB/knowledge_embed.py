import json
import pandas as pd
import mysql.connector
import certifi
import streamlit as st

from sentence_transformers import SentenceTransformer

# Instance Embedder

embedder = SentenceTransformer('BAAI/bge-m3')

# Kredensial dibaca dari .streamlit/secrets.toml (file ini JANGAN di-commit ke git)
db = mysql.connector.connect(
  host = st.secrets["TIDB_HOST"],
  port = st.secrets.get("TIDB_PORT", 4000),
  user = st.secrets["TIDB_USER"],
  password = st.secrets["TIDB_PASSWORD"],
  database = st.secrets.get("TIDB_DATABASE", "RAG"),
  ssl_ca = certifi.where(),
  ssl_verify_cert = True,
  ssl_verify_identity = True
)

curr = db.cursor()

# NOTE: tabel DOCS perlu punya kolom `source` (TEXT/VARCHAR).
# Kalau belum ada, jalankan dulu sekali di TiDB:
#   ALTER TABLE DOCS ADD COLUMN source TEXT;

# Baca data Knowledge
df = pd.read_csv("knowledge_base_final.csv")
# print(df)

for index, row in df.iterrows():
        # Pakai formatted_text (sudah ada format "Pertanyaan: ...\nJawaban: ...")
        # biar konsisten sama extract_answer_only() di app Streamlit
        text = str(row['formatted_text'])
        source = str(row['source'])

        try:
                embedding_list = embedder.encode(text).tolist()
                embedding_str = json.dumps(embedding_list)

                sql_query = """
                                INSERT INTO DOCS (text, source, embedding) VALUES (%s, %s, %s)
                        """

                curr.execute(sql_query, (text, source, embedding_str))
                print(f"data index-{index} berhasil ditambah")
        except Exception as e:
                print(f"error: {e}")
                print(f"data index-{index} gagal ditambah")

db.commit()
curr.close()
print("Data berhasil ditambah")