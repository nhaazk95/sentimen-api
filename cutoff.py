import pandas as pd

df = pd.read_csv("data_gabungan_final.csv")
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)

# Filter sampai 31 Agustus 2026 (23:59:59)
df = df[df['tanggal_ulasan'] <= '2026-08-31 23:59:59']

# Hitung total ulasan unik
total = len(df.drop_duplicates(subset=['rs', 'user', 'teks']))
print(f"Total ulasan unik (cutoff Agustus): {total}")

# Hitung ulasan sejak Januari 2025
df_2025 = df[df['tanggal_ulasan'] >= '2025-01-01']
total_2025 = len(df_2025.drop_duplicates(subset=['rs', 'user', 'teks']))
print(f"Ulasan sejak Januari 2025: {total_2025}")