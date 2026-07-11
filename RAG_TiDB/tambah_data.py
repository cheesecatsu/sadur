import mysql.connector, json, csv
from sentence_transformers import SentenceTransformer

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

model = SentenceTransformer('BAAI/bge-m3')
cursor = db.cursor()
berhasil = 0

# ── 1. Masukkan 52 data dari CSV ──────────────────────────────
with open('knowledge_base_final.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        text = row['formatted_text']
        emb  = model.encode(text, normalize_embeddings=True).tolist()
        cursor.execute(
            "INSERT INTO docs (text, embedding) VALUES (%s, %s)",
            (text, json.dumps(emb))
        )
        berhasil += 1
        print(f"✅ CSV [{berhasil}]: {row['question'][:50]}")

# ── 2. Tambah data teks eksposisi ─────────────────────────────
data_baru = [
    ("Apa struktur teks eksposisi?",
     "Teks eksposisi memiliki 3 bagian utama sesuai Kurikulum Merdeka: (1) Pernyataan Umum/Tesis — berisi pendirian atau sudut pandang penulis terhadap topik, (2) Argumentasi — berisi alasan, fakta, dan bukti yang mendukung tesis, (3) Penegasan Ulang — berisi simpulan yang memperkuat kembali tesis di bagian awal."),
    ("Apa ciri-ciri teks eksposisi?",
     "Ciri teks eksposisi: (1) bersifat informatif dan objektif, (2) menggunakan fakta dan data sebagai pendukung argumen, (3) menggunakan kata denotatif, (4) strukturnya terdiri dari tesis, argumentasi, dan penegasan ulang."),
    ("Apa itu tesis dalam teks eksposisi?",
     "Tesis adalah bagian pembuka teks eksposisi yang berisi pernyataan umum atau pendirian penulis terhadap suatu topik. Tesis harus jelas dan dapat dikembangkan menjadi argumen-argumen pada bagian berikutnya."),
    ("Apa itu argumentasi dalam teks eksposisi?",
     "Argumentasi adalah bagian isi teks eksposisi yang berisi alasan, fakta, dan bukti yang mendukung tesis penulis. Semakin kuat argumentasinya, semakin meyakinkan teks eksposisi tersebut."),
    ("Apa itu penegasan ulang dalam teks eksposisi?",
     "Penegasan ulang adalah bagian penutup teks eksposisi yang berisi simpulan yang memperkuat kembali tesis. Biasanya diawali dengan kata seperti 'dengan demikian', 'jadi', atau 'oleh karena itu'."),
    ("Apa bedanya teks eksposisi dan teks argumentasi?",
     "Teks eksposisi bertujuan memaparkan informasi secara objektif dengan struktur tesis-argumentasi-penegasan ulang. Teks argumentasi lebih bertujuan meyakinkan pembaca. Eksposisi lebih bersifat menjelaskan, argumentasi lebih bersifat membujuk."),
]

for q, a in data_baru:
    text = f"Pertanyaan: {q}\nJawaban: {a}"
    emb  = model.encode(text, normalize_embeddings=True).tolist()
    cursor.execute(
        "INSERT INTO docs (text, embedding) VALUES (%s, %s)",
        (text, json.dumps(emb))
    )
    berhasil += 1
    print(f"✅ Eksposisi: {q[:50]}")

db.commit()
cursor.close()
db.close()
print(f"\nTotal berhasil: {berhasil} baris masuk ke tabel docs")