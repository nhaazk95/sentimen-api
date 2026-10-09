import csv
import re
import sys

import pandas as pd

PATH = "data/training/train_pool.csv"
AUDIT = "audit_label.csv"
norm = lambda s: " ".join(str(s).lower().split())

POLA = {
    "balasan_pemilik": r"\byth\b|atas ulasan dan rating|kami ucapkan|manajemen kami|kami berkomitmen",
    "ucapan_doa": r"terima ?kasih|trimakasih|makasih|suwun|semoga|amanah|barakallah|aamiin|\bamin\b",
    "pertanyaan": r"\?",
}


def kategori(teks):
    t = norm(teks)
    n = len(t.split())
    if re.search(POLA["balasan_pemilik"], t):
        return "balasan_pemilik"
    if re.search(POLA["pertanyaan"], t) and n <= 20:
        return "pertanyaan"
    if re.search(POLA["ucapan_doa"], t) and n <= 8:
        return "ucapan_doa"
    if n <= 3:
        return "sangat_pendek"
    return ""


df = pd.read_csv(PATH)

if "--terapkan" not in sys.argv:
    df["kategori"] = df["text"].apply(kategori)
    kand = df[df["kategori"] != ""]
    print(pd.crosstab(kand["kategori"], kand["label"]))
    out = kand[["kategori", "label", "koreksi", "text"]].rename(columns={"label": "label_sekarang"})
    out["label_baru"] = ["HAPUS" if k == "balasan_pemilik" else "" for k in out["kategori"]]
    out.sort_values(["kategori", "label_sekarang"]).to_csv(AUDIT, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"\n{len(out)} kandidat disimpan di {AUDIT}. Isi kolom label_baru "
          f"(Positif/Netral/Negatif, atau HAPUS), lalu jalankan: python audit_label.py --terapkan")
else:
    a = pd.read_csv(AUDIT).fillna("")
    a = a[a["label_baru"].isin(["Positif", "Netral", "Negatif", "HAPUS"])]
    peta = {norm(t): l for t, l in zip(a["text"], a["label_baru"])}
    kunci = df["text"].apply(norm)
    hapus = kunci.map(peta) == "HAPUS"
    ganti = kunci.map(peta).isin(["Positif", "Netral", "Negatif"])
    df.loc[ganti, "label"] = kunci[ganti].map(peta)
    df.loc[ganti, "koreksi"] = 1
    df = df[~hapus]
    df.to_csv(PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Diganti: {int(ganti.sum())} baris | dihapus: {int(hapus.sum())} baris | total sekarang: {len(df)}")