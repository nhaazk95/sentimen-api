import pandas as pd

df = pd.read_csv("data_ulasan_compass.csv")
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)
df = df[df['tanggal_ulasan'] <= '2026-08-31 23:59:59']
df = df.drop_duplicates(subset=['rs', 'teks'], keep='first')

print(f"Total ulasan di data compass (cutoff Agustus): {len(df)}")