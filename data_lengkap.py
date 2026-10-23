import pandas as pd

# Gabung sampel + test_set
df_sampel = pd.read_csv("sampel_label.csv")
df_test = pd.read_csv("data/training/test_set.csv").rename(columns={"text": "Isi Ulasan", "label": "label_manual"})
df_all = pd.concat([df_sampel, df_test], ignore_index=True)
df_all.to_csv("data_ulasan_mentah.csv", index=False, encoding="utf-8-sig")
print(f"Tersimpan data_ulasan_mentah.csv: {len(df_all)} baris")