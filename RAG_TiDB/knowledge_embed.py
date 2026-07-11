import json
import pandas as pd
import mysql.connector

from sentence_transformers import SentenceTransformer

# Instance Embedder

embedder = SentenceTransformer('BAAI/bge-m3')

import mysql.connector

db = mysql.connector.connect(
  host = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
  port = 4000,
  user = "4Qz9gBVCQx5NSYQ.root",
  password = "IjuVu93yQQjJ65ND",
  database = "RAG",
  ssl_ca = "/etc/ssl/cert.pem",
  ssl_verify_cert = True,
  ssl_verify_identity = True
)

curr = db.cursor()

# Baca data Knowledge
df = pd.read_csv("knowledge_base_final.csv")
# print(df)

for index,row in df.iterrows():
        text = str(row['contoh_pertanyaan']) + "" + str(row['konten'])

        try:
                embedding_list = embedder.encode(text).tolist()
                embedding_str = json.dumps(embedding_list)

                sql_query = """
                                INSERT INTO DOCS (text,embedding) VALUES (%s,%s)
                        """

                curr.execute(sql_query, (text,embedding_str))
                print(f"data index-{index} berhasil ditambah")
        except Exception as e:
                print(f"error: {e}")
                print(f"data index-{index} gagal ditambah")

db.commit()
curr.close()
print("Data berhasil ditambah")