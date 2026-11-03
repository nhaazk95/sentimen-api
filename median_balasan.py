import pandas as pd

# Baca file CSV
df = pd.read_csv("data_ulasan_compass.csv")

# Konversi tanggal
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)

# Tanggal scraping (anggap hari ini, 30 Oktober 2026)
tanggal_scraping = pd.Timestamp("2026-10-30")

# Filter ulasan negatif (rating 1-2) yang sudah dibalas
negatif = df[df['rating'] <= 2].copy()
dibalas = negatif[negatif['balasan_pemilik'].notna() & (negatif['balasan_pemilik'].astype(str).str.strip() != "")].copy()

# Hitung selisih hari (estimasi maksimal)
dibalas['selisih_hari'] = (tanggal_scraping - dibalas['tanggal_ulasan']).dt.days

# Statistik
print(f"Total ulasan negatif yang dibalas: {len(dibalas)}")
print(f"Median estimasi waktu balasan (maksimal): {dibalas['selisih_hari'].median():.0f} hari")
print(f"Rata-rata estimasi waktu balasan (maksimal): {dibalas['selisih_hari'].mean():.0f} hari")
print(f"Estimasi tercepat: {dibalas['selisih_hari'].min()} hari")
print(f"Estimasi terlama: {dibalas['selisih_hari'].max()} hari")