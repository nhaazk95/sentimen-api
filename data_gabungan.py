import pandas as pd

# ==========================================
# 1. BACA FILE EXCEL (Data lama)
# ==========================================
print("Membaca Excel...")
f = pd.read_excel("Data_RS_Banyumas.xlsx")  # GANTI dengan path file Excel-mu
f = f[["Nama RS", "Username", "Rating", "Waktu Ulasan", "Isi Ulasan", "clean_text"]]
f.columns = ["rs", "user", "rating", "tanggal_ulasan", "teks", "clean"]
f["sumber"] = "file"

# ==========================================
# 2. BACA FILE CSV (Data baru)
# ==========================================
print("Membaca CSV...")
d = pd.read_csv("db_reviews.csv")  # GANTI dengan path file CSV-mu
d = d.rename(columns={
    "rs_name": "rs",
    "username": "user",
    "text": "teks",
    "published_at": "tanggal_ulasan",
    "clean_text": "clean"
})
d["sumber"] = "db"

# ==========================================
# 3. GABUNGKAN
# ==========================================
semua = pd.concat([f, d[["rs", "user", "rating", "tanggal_ulasan", "teks", "clean", "sumber"]]], ignore_index=True)

# ==========================================
# 4. KONVERSI TANGGAL & FILTER SAMPAI 31 AGUSTUS 2026
# ==========================================
semua['tanggal_ulasan'] = pd.to_datetime(semua['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)
semua = semua[semua['tanggal_ulasan'] <= '2026-08-31 23:59:59']

# ==========================================
# 5. HAPUS DUPLIKAT
# ==========================================
unik = semua.drop_duplicates(subset=['rs', 'user', 'teks'], keep='first')
print(f"\nTotal ulasan unik (cutoff Agustus 2026): {len(unik)}")

# ==========================================
# 6. ULASAN SEJAK JANUARI 2025
# ==========================================
df_2025 = unik[unik['tanggal_ulasan'] >= '2025-01-01']
print(f"Ulasan sejak Januari 2025: {len(df_2025)}")
print(f"Rata-rata per bulan (Jan 2025 - Agu 2026 = 20 bulan): {len(df_2025)/20:.0f}")

# ==========================================
# 7. RENTANG PER RS
# ==========================================
per_rs = unik.groupby('rs').size().sort_values()
print(f"\nJumlah ulasan per RS:")
print(per_rs)
print(f"Rentang: {per_rs.min()} hingga {per_rs.max()}")

# ==========================================
# 8. ULASAN YANG BERLABEL (untuk 82,0%)
# ==========================================
# Gabungkan dengan file yang sudah diprediksi (data_ulasan_compass.csv)
print("\nMembaca hasil prediksi...")
pred = pd.read_csv("data_ulasan_compass.csv")
pred['tanggal_ulasan'] = pd.to_datetime(pred['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)
pred = pred[pred['tanggal_ulasan'] <= '2026-08-31 23:59:59']

# Hapus duplikat
pred_unik = pred.drop_duplicates(subset=['rs', 'teks'], keep='first')
print(f"Total ulasan di data prediksi (cutoff Agustus): {len(pred_unik)}")

# ==========================================
# 9. SIMPAN HASIL
# ==========================================
unik.to_csv("data_gabungan_final.csv", index=False)
print("\nData gabungan disimpan ke data_gabungan_final.csv")