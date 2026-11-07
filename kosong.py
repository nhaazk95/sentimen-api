import pandas as pd
df = pd.read_csv("train.csv")
df["label_llm"] = pd.NA
df["label_final"] = pd.NA
df.to_csv("train.csv", index=False)
print("Label lama berhasil dikosongkan.")