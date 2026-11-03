import pandas as pd

d = pd.read_csv("label_288_asli.csv")   # sesuaikan nama file
d["tanggal_ulasan"] = pd.to_datetime(d["tanggal_ulasan"], errors="coerce")

print("Total baris:", len(d))
print("Tanggal paling awal :", d["tanggal_ulasan"].min())
print("Tanggal paling akhir:", d["tanggal_ulasan"].max())
print()
print("Distribusi per tahun:")
print(d["tanggal_ulasan"].dt.year.value_counts().sort_index())
print()
print("Distribusi per bulan:")
print(d["tanggal_ulasan"].dt.to_period("M").value_counts().sort_index())
print()
print("Distribusi rating:")
print(d["rating"].value_counts().sort_index())