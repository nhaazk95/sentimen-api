import pandas as pd
import joblib

# 1. Load data uji yang sudah kamu labeli manual
df = pd.read_csv("test2_fixed.csv")

# Pastikan kolom label manual tidak kosong
df = df[df['label'].notna() & (df['label'].astype(str).str.strip() != "")]
df['label_manual'] = df['label'].astype(str).str.lower().str.strip()

# 2. Load model terbaik yang baru saja kamu latih
# (Pastikan nama file ini sesuai dengan output dari script bandingkan_model.py sebelumnya)
model = joblib.load("model_terbaik_perbandingan.pkl")

# 3. Lakukan prediksi pada teks yang sudah dibersihkan (kolom 'clean')
X_test = df['clean'].astype(str)
df['prediksi_model'] = model.predict(X_test)

# 4. Cari yang prediksi modelnya BEDA dengan manual
salah = df[df['prediksi_model'] != df['label_manual']].copy()

# 5. Hitung statistik
total_uji = len(df)
jumlah_salah = len(salah)
akurasi = (1 - (jumlah_salah / total_uji)) * 100 if total_uji > 0 else 0

print(f"Total data uji        : {total_uji} ulasan")
print(f"Tebakan model salah   : {jumlah_salah} ulasan")
print(f"Akurasi pada data uji : {akurasi:.2f}%")

# 6. Tampilkan pola kesalahan
print("\n=== POLA KESALAHAN (Tebakan Model -> Padahal Manualnya) ===")
pola_salah = salah.groupby(['prediksi_model', 'label_manual']).size().reset_index(name='Jumlah')
pola_salah = pola_salah.sort_values(by='Jumlah', ascending=False)
print(pola_salah.to_string(index=False))

# 7. Simpan ke CSV untuk dibaca di Excel
kolom_penting = ['rs_name', 'rating', 'text', 'clean', 'prediksi_model', 'label_manual']
# (Sesuaikan nama kolom 'rs_name' dan 'text' dengan yang ada di CSV-mu)
salah[kolom_penting].to_csv("kesalahan_model_data_uji.csv", index=False, encoding="utf-8-sig")

print("\n-> File 'kesalahan_model_data_uji.csv' berhasil dibuat.")
print("   Buka file tersebut di Excel untuk melihat ulasan mana yang salah tebak.")