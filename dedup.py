import csv
import pandas as pd

norm = lambda s: s.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
df = pd.read_csv("data/training/train_pool.csv")
print("Sebelum:", len(df))

df["_k"] = norm(df["text"])
df = (df.sort_values("koreksi", ascending=False)
        .drop_duplicates("_k", keep="first")
        .drop(columns="_k")
        .sort_index())
df.to_csv("data/training/train_pool.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
print("Sesudah:", len(df), "| koreksi=1:", int(df["koreksi"].sum()))
print(df[df["koreksi"] == 1]["label"].value_counts().to_dict())