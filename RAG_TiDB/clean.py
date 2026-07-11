import pandas as pd
import re

# Baca CSV
df = pd.read_csv("knowledge_base_final.csv")

def clean_answer(text):
    """Hilangkan pertanyaan ganda di awal jawaban"""
    # Pisahkan kalimat pertama
    sentences = text.split('. ')
    
    # Kalo kalimat pertama mengandung "?", skip
    if '?' in sentences[0]:
        # Ambil dari kalimat kedua
        return '. '.join(sentences[1:])
    return text

def clean_question(text):
    """Bersihkan pertanyaan"""
    text = str(text).strip()
    # Kalo ada "?" di tengah, ambil bagian pertama aja
    if '?' in text:
        parts = text.split('?')
        return parts[0] + '?'
    return text

# Bersihkan
for idx, row in df.iterrows():
    df.loc[idx, 'question'] = clean_question(row['question'])
    df.loc[idx, 'answer'] = clean_answer(row['answer'])
    # Update formatted_text
    df.loc[idx, 'formatted_text'] = f"P: {df.loc[idx, 'question']}\nJ: {df.loc[idx, 'answer']}"

# Simpan
df.to_csv("knowledge_final_fix.csv", index=False)
print(f"✅ {len(df)} data berhasil dibersihkan!")
print("\n📊 Preview:")
print(df[['question', 'answer']].head(3))