import csv
import pandas as pd

PATH = "train_pool.csv"

koreksi = [
    ("pelayanan sangat bagus tidak ada kekurangan", "Positif"),
    ("Perfect", "Positif"),
    ("Bagus tempatnya", "Positif"),
    ("baik", "Positif"),
    ("Pelayanan di igd sudah baik", "Positif"),
    ("radiologi baik", "Positif"),
    ("Langganan bunda sudah lama,pasien dokter Arinton ruang melati.pelayanan dokter dan perawat baik.anak cucu sudah langganan rsu bunda lamaa.mks RS bunda", "Positif"),
    ("Sya pasien de sutrisno ibu raniyah sya puasa dg pelyannyaaa", "Positif"),
]

df = pd.read_csv(PATH)
if "koreksi" not in df.columns:
    df["koreksi"] = 0
df["koreksi"] = df["koreksi"].fillna(0).astype(int)

baru = pd.DataFrame(koreksi, columns=["text", "label"])
baru["koreksi"] = 1

norm = lambda s: s.str.strip().str.lower()
kunci = dict(zip(norm(baru["text"]), baru["label"]))

# 1. Yang sudah ada: set label benar + tandai koreksi
ada = norm(df["text"]).isin(kunci)
df.loc[ada, "label"] = norm(df.loc[ada, "text"]).map(kunci)
df.loc[ada, "koreksi"] = 1
print(f"Ditandai koreksi: {ada.sum()} baris")

# 2. Yang belum ada: tambahkan sebagai baris baru
belum = ~norm(baru["text"]).isin(set(norm(df["text"])))
df = pd.concat([df, baru[belum]], ignore_index=True)
print(f"Ditambahkan baru: {belum.sum()} baris")

df.to_csv(PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(df["label"].value_counts().to_dict(), "| koreksi=1:", int(df["koreksi"].sum()))