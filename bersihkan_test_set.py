import csv
import pandas as pd

norm = lambda s: s.astype(str).str.strip().str.lower()
train = pd.read_csv("data/training/train_pool.csv")
test = pd.read_csv("data/training/test_set.csv")

kor = set(norm(train.loc[train["koreksi"] == 1, "text"]))
hapus = norm(test["text"]).isin(kor)
print(f"Dihapus dari test_set: {hapus.sum()} dari {len(test)} baris")

test[~hapus].to_csv("data/training/test_set.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)