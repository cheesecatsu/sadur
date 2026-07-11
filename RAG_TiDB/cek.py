import mysql.connector

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

cursor = db.cursor()

# Cek struktur tabel docs
cursor.execute("DESCRIBE docs")
print("=== STRUKTUR TABEL docs ===")
for row in cursor.fetchall():
    print(row)

print("\n")

# Cek isi tabel (5 data pertama)
cursor.execute("SELECT * FROM docs LIMIT 5")
print("=== 5 DATA PERTAMA ===")
for row in cursor.fetchall():
    print(row)

cursor.close()
db.close()