import pandas as pd
import joblib
from preprocessing import full_preprocess, load_slang_dict

# 1. Load model
pipeline = joblib.load('svm_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')
slang_dict = load_slang_dict('slang_dict.json')

def preprocess(text):
    return full_preprocess(text, slang_dict)

# 2. Baca data prediksi (compass)
df = pd.read_csv("data_ulasan_compass.csv")
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)
df = df[df['tanggal_ulasan'] <= '2026-08-31 23:59:59']

# 3. Filter rating 4-5
tinggi = df[df['rating'] >= 4].copy()
print(f"Total ulasan rating 4-5: {len(tinggi)}")

# 4. Prediksi
clean_texts = tinggi['teks'].fillna("").apply(preprocess)
pred_angka = pipeline.predict(clean_texts)
tinggi['prediksi'] = label_encoder.inverse_transform(pred_angka)

# 5. Filter yang diprediksi NEGATIF
negatif_tinggi = tinggi[tinggi['prediksi'] == 'Negatif'].copy()
print(f"Ulasan rating 4-5 yang diprediksi NEGATIF: {len(negatif_tinggi)}")

# 6. Simpan
negatif_tinggi[['rs', 'rating', 'tanggal_ulasan', 'teks', 'prediksi']].to_csv(
    "rating_tinggi_negatif.csv", index=False
)
print("Hasil disimpan ke rating_tinggi_negatif.csv")