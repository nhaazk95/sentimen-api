import pandas as pd
import joblib
from preprocessing import full_preprocess, load_slang_dict

pipeline = joblib.load('svm_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')
slang_dict = load_slang_dict('slang_dict.json')

def preprocess(text):
    return full_preprocess(text, slang_dict)

df = pd.read_csv("data_gabungan_final.csv")
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)
df = df[df['tanggal_ulasan'] <= '2026-08-31 23:59:59']

tinggi = df[df['rating'] >= 4].copy()
clean_texts = tinggi['teks'].fillna("").apply(preprocess)
pred_angka = pipeline.predict(clean_texts)
tinggi['prediksi'] = label_encoder.inverse_transform(pred_angka)

alarm = tinggi[tinggi['prediksi'] == 'Negatif'].copy()
print(f"Total ulasan rating 4-5 yang diprediksi negatif ([N]): {len(alarm)}")