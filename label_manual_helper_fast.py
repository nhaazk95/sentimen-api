"""
label_manual_helper_fast.py
10 baris per layar, ketik 10 karakter (p/n/t), Enter.
Auto-save tiap batch. Bisa dilanjutkan.
"""
import pandas as pd
import os

FILE_IN = "sampel_label.csv"
FILE_OUT = "sampel_label_terisi.csv"
BATCH = 10
VALID = {"p": "Positif", "n": "Negatif", "t": "Netral"}

if os.path.exists(FILE_OUT):
    df = pd.read_csv(FILE_OUT, encoding="utf-8-sig")
else:
    df = pd.read_csv(FILE_IN, encoding="utf-8-sig")
    df["label_manual"] = ""

def is_labeled(v):
    return str(v).strip().capitalize() in ("Positif", "Negatif", "Netral")

# Kumpulkan baris yang belum berlabel
todo = [i for i in df.index if not is_labeled(df.at[i, "label_manual"])]
print(f"Belum berlabel: {len(todo)}/{len(df)}\n")

i = 0
while i < len(todo):
    batch_idx = todo[i:i+BATCH]
    print("=" * 78)
    for j, idx in enumerate(batch_idx, 1):
        row = df.loc[idx]
        txt = str(row["Isi Ulasan"]).replace("\n", " ")[:160]
        print(f"{j:2d}. [{row['Rating']}] {txt}")
    print("=" * 78)
    ans = input(f"Label {len(batch_idx)} baris (p/n/t, spasi boleh; s=skip batch; q=quit) > ").strip().lower()
    if ans == "q":
        break
    if ans == "s":
        i += BATCH
        continue
    chars = [c for c in ans if c in VALID]
    if len(chars) != len(batch_idx):
        print(f"[!] Butuh {len(batch_idx)} karakter, dapat {len(chars)}. Ulangi batch ini.\n")
        continue
    for idx, c in zip(batch_idx, chars):
        df.at[idx, "label_manual"] = VALID[c]
    df.to_csv(FILE_OUT, index=False, encoding="utf-8-sig")
    i += BATCH
    print(f"[auto-save] {i}/{len(todo)} diproses\n")

df.to_csv(FILE_OUT, index=False, encoding="utf-8-sig")
print("\nSelesai.")
print("Sebaran:", df["label_manual"].value_counts().to_dict())
print(f"File: {FILE_OUT}")