import streamlit as st
import mysql.connector
import json
import re
import certifi
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
        ssl_ca=certifi.where(),
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

# ===== FUNGSI SEARCH (DENGAN SUMBER UNTUK DITAMPILKAN) =====
def search_document(query, k_top=3, max_distance=0.6):
    db = get_live_db()
    curr = db.cursor()

    query_embedding = embedder.encode(query).tolist()
    query_embedding_str = json.dumps(query_embedding)

    sql_query = f"""
        SELECT text, source, vec_cosine_distance(embedding, %s) AS distance
        FROM DOCS
        HAVING distance <= %s
        ORDER BY distance ASC
        LIMIT {k_top}
    """

    curr.execute(sql_query, (query_embedding_str, max_distance))
    results = curr.fetchall()
    curr.close()

    return [{"text": text, "source": source, "distance": distance} for text, source, distance in results]

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
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu asisten yang SELALU menjawab dalam Bahasa Indonesia, apa pun bahasa yang dipakai user (termasuk sapaan singkat seperti 'hi'/'hello'). Jangan pernah membalas dalam Bahasa Inggris."
                    },
                    {"role": "user", "content": query}
                ],
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

    prompt = f"""Kamu asisten yang SELALU menjawab dalam Bahasa Indonesia, apa pun bahasa pertanyaan user. Jawab PERMINTAAN USER pakai ATURAN EYD di bawah kalau relevan (boleh sebagian). Kalau kalimat user sudah benar, bilang begitu. Kalau kalimat perlu diperbaiki, tulis versi perbaikannya + alasan singkat (1-2 kalimat). Kalau tidak ada aturan yang relevan sama sekali atau USER bertanya di luar lingkup, bilang: "Maaf, informasi ini di luar cakupan materi yang saya miliki." Jangan tampilkan proses berpikir, langsung jawaban akhir saja.

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

BOT_AVATAR = "https://raw.githubusercontent.com/cheesecatsu/sadur/refs/heads/main/RAG_TiDB/assets/bot-avatar.svg"
USER_AVATAR = "https://raw.githubusercontent.com/cheesecatsu/sadur/refs/heads/main/RAG_TiDB/assets/user-avatar.svg" 

# ===== CUSTOM CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&display=swap');

h1 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
}

/* Bubble chat lebih membulat & ada sedikit shadow */
.stChatMessage {
    border-radius: 16px;
    padding: 4px 6px;
}

/* Bubble user dikasih warna beda biar kontras sama bot */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background-color: #FFF4EE;
}

/* Card sumber */
.sumber-card {
    background-color: #FFF4EE;
    border-left: 4px solid #E8622C;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.9rem;
}

/* Tombol sidebar */
.stButton > button {
    border-radius: 10px;
    border: 1px solid #E8622C;
}
</style>
""", unsafe_allow_html=True)

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


def render_sumber(sumber_docs):
    """Tampilkan daftar sumber unik yang dipakai sebagai card, tanpa skor jarak."""
    with st.expander("📄 Sumber yang dipakai"):
        seen = set()
        for s in sumber_docs:
            source_name = s.get("source") or "Sumber tidak diketahui"
            if source_name in seen:
                continue
            seen.add(source_name)
            st.markdown(
                f'<div class="sumber-card">📚 {source_name}</div>',
                unsafe_allow_html=True
            )


# Tampilkan chat
for message in st.session_state.messages:
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sumber"):
            render_sumber(message["sumber"])

# Input
if prompt := st.chat_input("Ketik pertanyaanmu..."):
    # Pesan user
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Ambil dokumen sumber duluan biar bisa ditampilkan setelah jawaban selesai
    try:
        sumber_docs = search_document(prompt, k_top=3)
    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        sumber_docs = []

    # Respon bot (dengan streaming)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("🔍 Mencari jawaban..."):
            try:
                response_stream = jawab_stream(prompt)
                response_text = st.write_stream(response_stream)

                if sumber_docs:
                    render_sumber(sumber_docs)

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