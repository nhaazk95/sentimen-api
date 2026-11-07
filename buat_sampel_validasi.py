# buat_sampel_validasi.py
import pandas as pd

df = pd.read_csv("data_gabungan_final.csv")
df = df[df["label_llm"].notna()].copy()
df["tanggal_ulasan"] = pd.to_datetime(df["tanggal_ulasan"], errors="coerce")

# Sampel acak 100 (proporsional seperti distribusi asli)
acak = df.sample(n=100, random_state=42)

# Tambahan khusus: 50 negatif + 40 netral (agar kelas kecil terukur)
neg = df[df["label_llm"] == "negatif"].sample(n=50, random_state=42)
net = df[df["label_llm"] == "netral"].sample(n=40, random_state=42)

sampel = pd.concat([acak, neg, net]).drop_duplicates(subset=["rs", "teks"]).reset_index(drop=True)
sampel["id_sampel"] = range(len(sampel))

# Sembunyikan label_llm dulu supaya reviewer tidak terpengaruh
out = sampel[["id_sampel", "rs", "tanggal_ulasan", "clean"]].copy()
out["label_manual"] = ""

out.to_csv("sampel_validasi.csv", index=False)
sampel[["id_sampel", "label_llm"]].to_csv("kunci_label_llm.csv", index=False)

print(f"Sampel: {len(out)} baris -> sampel_validasi.csv")
print(f"Kunci LLM disimpan terpisah -> kunci_label_llm.csv")
print("\nIsi 'label_manual' (positif/netral/negatif) di sampel_validasi.csv")
print("JANGAN buka kunci_label_llm.csv sampai selesai mengisi.")