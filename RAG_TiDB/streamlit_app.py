import streamlit as st
import mysql.connector
import json
import re
from groq import Groq, RateLimitError, APIConnectionError, APIStatusError, AuthenticationError, APITimeoutError
from sentence_transformers import SentenceTransformer

# ===== KONFIGURASI =====
GROQ_MODEL = "llama-3.3-70b-versatile"

# ===== LOAD MODEL =====
@st.cache_resource
def load_models():
    print("📥 Memuat model...")
    llm = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedder = SentenceTransformer('BAAI/bge-m3')
    print("✅ Model siap!\n")
    return llm, embedder

llm_agent, embedder = load_models()

# ===== KONEKSI DATABASE (RAG - TiDB Cloud) =====
@st.cache_resource
def get_db():
    """Koneksi database untuk RAG. Di-cache biar gak connect ulang tiap rerun."""
    return mysql.connector.connect(
        host=st.secrets["TIDB_HOST"],
        port=st.secrets.get("TIDB_PORT", 4000),
        user=st.secrets["TIDB_USER"],
        password=st.secrets["TIDB_PASSWORD"],
        database=st.secrets.get("TIDB_DATABASE", "RAG"),
        ssl_ca="/etc/ssl/certs/ca-certificates.crt",
        ssl_verify_cert=True,
        ssl_verify_identity=True
    )

def get_live_db():
    """
    Pastikan koneksi masih hidup sebelum dipakai.
    Kalau koneksi mati/timeout, ping(reconnect=True) otomatis bikin ulang.
    Ini menggantikan pendekatan lama (koneksi dibiarkan terbuka tanpa dicek),
    yang gampang error kalau koneksi drop karena idle terlalu lama.
    """
    db = get_db()
    try:
        db.ping(reconnect=True, attempts=3, delay=1)
    except mysql.connector.Error:
        # Kalau ping tetap gagal, hapus cache biar dibuat koneksi baru dari nol
        get_db.clear()
        db = get_db()
    return db

# ===== FUNGSI SEARCH (DENGAN SKOR JARAK UNTUK DITAMPILKAN) =====
def search_document(query, k_top=3, max_distance=0.6):
    db = get_live_db()
    curr = db.cursor()

    query_embedding = embedder.encode(query).tolist()
    query_embedding_str = json.dumps(query_embedding)

    sql_query = f"""
        SELECT text, vec_cosine_distance(embedding, %s) AS distance
        FROM DOCS
        HAVING distance <= %s
        ORDER BY distance ASC
        LIMIT {k_top}
    """

    curr.execute(sql_query, (query_embedding_str, max_distance))
    results = curr.fetchall()
    curr.close()

    return [{"text": text, "distance": distance} for text, distance in results]

def extract_answer_only(text):
    """Buang prefix 'Pertanyaan: ...' dari chunk, sisain isi Jawaban-nya aja,
    biar LLM gak ketuker antara pertanyaan di dalam context vs pertanyaan user."""
    match = re.search(r"Jawaban:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text


def handle_groq_error(e):
    """Terjemahin error dari Groq API jadi pesan yang enak dibaca user,
    sambil tetep nge-print detail aslinya ke terminal/log buat debug."""
    print(f"[GROQ ERROR] {type(e).__name__}: {e}")

    if isinstance(e, RateLimitError):
        yield "⏳ Lagi banyak yang nanya nih, server AI-nya kepenuhan permintaan. Coba tunggu 30-60 detik lalu tanya lagi ya."
    elif isinstance(e, AuthenticationError):
        yield "🔑 Ada masalah konfigurasi API key. Tolong beri tahu pengelola aplikasi (bukan salah kamu)."
    elif isinstance(e, APITimeoutError):
        yield "⌛ Server AI lambat merespons. Coba kirim ulang pertanyaannya."
    elif isinstance(e, APIConnectionError):
        yield "📡 Gagal terhubung ke server AI. Cek koneksi internet, atau coba lagi sebentar lagi."
    elif isinstance(e, APIStatusError):
        yield f"⚠️ Server AI sedang bermasalah (kode {e.status_code}). Coba lagi beberapa saat lagi."
    else:
        yield "❌ Terjadi kesalahan tak terduga. Coba lagi, atau hubungi pengelola aplikasi kalau berulang."


# ===== FUNGSI JAWAB DENGAN STREAMING (GROQ) =====
def jawab_stream(query):
    try:
        docs = search_document(query, k_top=3, max_distance=0.6)
    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        yield "⚠️ Gagal terhubung ke database pengetahuan. Coba lagi sebentar lagi."
        return

    # Kalo ga ada dokumen relevan
    if not docs:
        try:
            stream = llm_agent.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": query}],
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except (RateLimitError, APIStatusError, APIConnectionError, APITimeoutError, AuthenticationError) as e:
            yield from handle_groq_error(e)
        return

    # Kalo ada dokumen relevan: ambil isi Jawaban-nya aja, buang "Pertanyaan: ..."
    # biar LLM gak ketuker antara pertanyaan di dalam context vs pertanyaan user
    context = "\n\n".join(extract_answer_only(d["text"]) for d in docs)

    prompt = f"""Kamu asisten Bahasa Indonesia. Jawab PERMINTAAN USER pakai ATURAN EYD di bawah kalau relevan (boleh sebagian). Kalau kalimat user sudah benar, bilang begitu. Kalau kalimat perlu diperbaiki, tulis versi perbaikannya + alasan singkat (1-2 kalimat). Kalau tidak ada aturan yang relevan sama sekali atau USER bertanya di luar lingkup, bilang: "Maaf, informasi ini di luar cakupan materi yang saya miliki." Jangan tampilkan proses berpikir, langsung jawaban akhir saja.

ATURAN EYD:
{context}

PERMINTAAN USER: {query}

JAWABAN:"""

    print("=" * 50)
    print("PROMPT YANG DIKIRIM:")
    print(prompt)
    print("=" * 50)

    try:
        stream = llm_agent.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=400  # cukup buat jawaban + penjelasan singkat, biasanya gak butuh lebih
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except (RateLimitError, APIStatusError, APIConnectionError, APITimeoutError, AuthenticationError) as e:
        yield from handle_groq_error(e)


# ==========================================
# ===== UI STREAMLIT =====
# ==========================================

st.set_page_config(
    page_title="Sadur AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Sadur AI")
st.caption("Tanyakan apa saja tentang teks eksposisi, esai, atau topik lainnya!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! 👋 Ada yang bisa saya bantu?"}
    ]

with st.sidebar:
    if st.button("🔄 Mulai Percakapan Baru"):
        st.session_state.pop("messages", None)
        st.rerun()

# Tampilkan chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sumber"):
            with st.expander("📄 Sumber yang dipakai"):
                for s in message["sumber"]:
                    st.caption(f"Skor jarak: {s['distance']:.4f}")
                    st.write(s["text"])
                    st.divider()

# Input
if prompt := st.chat_input("Ketik pertanyaanmu..."):
    # Pesan user
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Ambil dokumen sumber duluan biar bisa ditampilkan setelah jawaban selesai
    try:
        sumber_docs = search_document(prompt, k_top=3)
    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        sumber_docs = []

    # Respon bot (dengan streaming)
    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari jawaban..."):
            try:
                response_stream = jawab_stream(prompt)
                response_text = st.write_stream(response_stream)

                if sumber_docs:
                    with st.expander("📄 Sumber yang dipakai"):
                        for s in sumber_docs:
                            st.caption(f"Skor jarak: {s['distance']:.4f}")
                            st.write(s["text"])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sumber": sumber_docs
                })
            except Exception as e:
                # Jaring pengaman terakhir buat error yang bener-bener gak terduga
                error_msg = "❌ Maaf, ada kesalahan tak terduga. Coba lagi atau hubungi pengelola aplikasi."
                st.error(error_msg)
                print(f"[UNEXPECTED ERROR] {type(e).__name__}: {e}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})