import pandas as pd

# Baca data
df = pd.read_csv("data_gabungan_final.csv")
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)

# Filter: Juli 2024 - Agustus 2026, rating <= 2
per = df[(df['tanggal_ulasan'] >= '2024-07-01') & (df['tanggal_ulasan'] < '2026-09-01') & (df['rating'] <= 2)]

print(f"Total ulasan negatif (Juli 2024 - Agustus 2026): {len(per)}")