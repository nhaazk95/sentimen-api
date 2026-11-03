import pandas as pd

# Baca file
df = pd.read_csv("data_ulasan_compass.csv")
df['tanggal_ulasan'] = pd.to_datetime(df['tanggal_ulasan'], errors='coerce').dt.tz_localize(None)

# Batas tanggal: sampai 31 Agustus 2026
df = df[df['tanggal_ulasan'] <= '2026-08-31']

# Definisi negatif: rating <= 2
negatif = df[df['rating'] <= 2].copy()

# Kolom balasan
def ada_balasan(x):
    return pd.notna(x) and str(x).strip() != ""

negatif['dibalas'] = negatif['balasan_pemilik'].apply(ada_balasan)

# Analisis 1: Buang ulasan yang terbit <30 hari sebelum pengambilan data (30 Okt 2026)
# Artinya, hanya ulasan yang terbit <= 30 Sep 2026
df_30hari = negatif[negatif['tanggal_ulasan'] <= '2026-09-30']

total_30 = len(df_30hari)
dibalas_30 = df_30hari['dibalas'].sum()
tidak_dibalas_30 = total_30 - dibalas_30

print("=== ANALISIS 1: Buang ulasan <30 hari ===")
print(f"Total negatif (<= 30 Sep 2026): {total_30}")
print(f"Dibalas: {dibalas_30} ({dibalas_30/total_30*100:.1f}%)")
print(f"Tidak dibalas: {tidak_dibalas_30} ({tidak_dibalas_30/total_30*100:.1f}%)")

# Analisis 2: Bandingkan rating <=2 vs rating >2 (tapi teks mengandung keluhan)
# Karena kita tidak punya label model di CSV ini, kita asumsikan rating <=2 sebagai "negatif rating"
# dan rating 3-5 sebagai "positif rating". Tapi karena tidak ada prediksi model, kita tidak bisa bandingkan.
# Sebagai gantinya, kita hanya bisa bandingkan rating <=2 vs rating 3.

# Analisis rating 3 (netral)
rating3 = df[df['rating'] == 3].copy()
rating3['dibalas'] = rating3['balasan_pemilik'].apply(ada_balasan)
total_r3 = len(rating3)
dibalas_r3 = rating3['dibalas'].sum()
print("\n=== ANALISIS 2: Rating 3 (Netral) ===")
print(f"Total rating 3: {total_r3}")
print(f"Dibalas: {dibalas_r3} ({dibalas_r3/total_r3*100:.1f}%)")

# Analisis rating 4-5 (positif)
rating45 = df[df['rating'] >= 4].copy()
rating45['dibalas'] = rating45['balasan_pemilik'].apply(ada_balasan)
total_r45 = len(rating45)
dibalas_r45 = rating45['dibalas'].sum()
print("\n=== ANALISIS 3: Rating 4-5 (Positif) ===")
print(f"Total rating 4-5: {total_r45}")
print(f"Dibalas: {dibalas_r45} ({dibalas_r45/total_r45*100:.1f}%)")