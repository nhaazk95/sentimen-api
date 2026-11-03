import pandas as pd
import joblib
from preprocessing import full_preprocess, load_slang_dict

# Load model & tools versi SKRIPSI
pipeline     = joblib.load('svm_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')
slang_dict   = load_slang_dict('slang_dict.json')

def preprocess(text):
    return full_preprocess(text, slang_dict)

df = pd.read_csv("data_gabungan_final.csv")
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)
df = df[df['tanggal_ulasan'] <= '2026-08-31 23:59:59']

print(f"Baris setelah filter tanggal: {len(df)}")

# Filter rating >= 4 (perhatikan: TIDAK dibatasi <= 5)
tinggi = df[df['rating'] >= 4].copy()
print(f"Baris rating >= 4: {len(tinggi)}")

# Preprocess & prediksi
clean_texts = tinggi['teks'].fillna("").apply(preprocess)
pred_angka  = pipeline.predict(clean_texts)
tinggi['prediksi'] = label_encoder.inverse_transform(pred_angka)

print("\nDistribusi prediksi pada rating >= 4:")
print(tinggi['prediksi'].value_counts())
print()

# Filter negatif
alarm = tinggi[tinggi['prediksi'] == 'Negatif'].copy()
print(f">>> Total rating >= 4 yang diprediksi Negatif: {len(alarm)}")

# Simpan untuk labeling
alarm[["rs", "rating", "tanggal_ulasan", "teks"]].assign(label_manual="") \
    .to_csv("label_288_asli.csv", index=False, encoding="utf-8-sig")
print("-> label_288_asli.csv")