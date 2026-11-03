import pandas as pd
import joblib
from preprocessing import full_preprocess, load_slang_dict

pipeline      = joblib.load('svm_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')
slang_dict    = load_slang_dict('slang_dict.json')

def preprocess(text):
    return full_preprocess(text, slang_dict)

df = pd.read_csv("data_gabungan_final.csv")
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)
df = df[df['tanggal_ulasan'] <= '2026-08-31 23:59:59']

# Prediksi SEMUA ulasan
clean_texts = df['teks'].fillna("").apply(preprocess)
pred_angka  = pipeline.predict(clean_texts)
df['prediksi'] = label_encoder.inverse_transform(pred_angka)

# Populasi: semua prediksi Negatif (semua rating)
neg = df[df['prediksi'] == 'Negatif'].copy()
print(f"Total prediksi Negatif (semua rating): {len(neg)}")

# Sampel 60
sampel = neg.sample(n=min(60, len(neg)), random_state=42)
sampel[["rs", "rating", "tanggal_ulasan", "teks"]].assign(label_manual="") \
    .to_csv("label_negatif_60.csv", index=False, encoding="utf-8-sig")
print(f"-> label_negatif_60.csv ({len(sampel)} baris)")