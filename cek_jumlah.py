import pandas as pd

df = pd.read_csv("data/training/train_pool.csv")
print("koreksi=1:", int(df["koreksi"].sum()))
print(df[df["koreksi"] == 1]["label"].value_counts().to_dict())
norm = lambda s: s.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
print("Duplikat teks:", int(norm(df["text"]).duplicated().sum()))