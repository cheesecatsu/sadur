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

curr = db.cursor()

# Hapus semua data
curr.execute("DELETE FROM DOCS")
db.commit()

print("🗑️ Data lama dihapus!")

curr.close()
db.close()