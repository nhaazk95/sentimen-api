import pandas as pd

d = pd.read_csv("data_gabungan_final_with_pred.csv")
d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
d["tanggal_ulasan"] = pd.to_datetime(d["tanggal_ulasan"], errors="coerce").dt.tz_localize(None)

mask_periode = (d["tanggal_ulasan"] >= "2024-07-01") & (d["tanggal_ulasan"] < "2026-09-01")

# Coba berbagai kombinasi
def cek(nama, mask):
    print(f"{nama:60s}: {mask.sum()}")

print("=== Cari kombinasi yang hasilnya 288 ===\n")

cek("Rating 4-5, negatif (semua periode)",
    (d.rating>=4)&(d.rating<=5)&(d.sentimen_kode==0))
cek("Rating 4-5, negatif (periode)",
    (d.rating>=4)&(d.rating<=5)&(d.sentimen_kode==0)&mask_periode)
cek("Rating 4-5, negatif+netral (semua)",
    (d.rating>=4)&(d.rating<=5)&(d.sentimen_kode.isin([0,1])))
cek("Rating 4-5, negatif+netral (periode)",
    (d.rating>=4)&(d.rating<=5)&(d.sentimen_kode.isin([0,1]))&mask_periode)
cek("Rating 3-5, negatif (semua)",
    (d.rating>=3)&(d.sentimen_kode==0))
cek("Rating 3-5, negatif (periode)",
    (d.rating>=3)&(d.sentimen_kode==0)&mask_periode)
cek("Rating >=4, negatif (semua)", (d.rating>=4)&(d.sentimen_kode==0))
cek("Rating >=4, negatif (periode)", (d.rating>=4)&(d.sentimen_kode==0)&mask_periode)
cek("Rating 4-5, apapun sentimennya (periode)",
    (d.rating>=4)&(d.rating<=5)&mask_periode)
cek("Negatif (semua periode)", d.sentimen_kode==0)
cek("Negatif (periode)", (d.sentimen_kode==0)&mask_periode)