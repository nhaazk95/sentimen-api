# hitung_validasi.py
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score

s = pd.read_csv("sampel_validasi.csv")
k = pd.read_csv("kunci_label_llm.csv")

m = s.merge(k, on="id_sampel")
m = m[m["label_manual"].notna() & (m["label_manual"].astype(str).str.strip() != "")]

print(f"Validasi pada {len(m)} sampel\n")
print(classification_report(m["label_manual"], m["label_llm"], digits=3, zero_division=0))
print("Cohen's kappa:", round(cohen_kappa_score(m["label_manual"], m["label_llm"]), 4))

# simpan yang salah
salah = m[m["label_manual"] != m["label_llm"]]
salah.to_csv("validasi_salah.csv", index=False)
print(f"\n{len(salah)} ketidaksepakatan -> validasi_salah.csv")