# buat_split_baru.py
# Menggabungkan train_fixed.csv + test2_fixed.csv, lalu membagi ulang (stratified per label)
# dengan aturan: teks yang sama TIDAK BOLEH ada di train dan uji sekaligus (anti-kebocoran).
# Hasil: train_v3.csv dan test_v3.csv (kolom `asal` mencatat asal tiap ulasan: train/uji).
import re, unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TRAIN_FILE, TEST_FILE = "train_v1.csv", "test_v1.csv"
RANDOM   = 42
N_SPLITS = 5          # 1 lipatan jadi uji -> uji ~20%

# ---------- baca & samakan kolom ----------
tr = pd.read_csv(TRAIN_FILE)
te = pd.read_csv(TEST_FILE)
te = te[~te["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)]   # bukan ulasan asli

A = pd.DataFrame(dict(rs=tr["rs"], rating=tr["rating"], teks=tr["teks"], label=tr["label_final"],
                      perlu_cek=tr["perlu_cek"], asal="train", catatan=tr["catatan"]))
B = pd.DataFrame(dict(rs=te["rs_name"], rating=te["rating"], teks=te["text"], label=te["label"],
                      perlu_cek=te["perlu_cek"], asal="uji", catatan=te["alasan"]))
P = pd.concat([A, B], ignore_index=True)

# ---------- kunci kelompok: teks identik (setelah dinormalkan) harus ikut satu sisi ----------
def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\\[rn]|<br>|[\r\n]+", " ", s)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()

P["kunci"] = P["teks"].map(norm)
kosong = P["kunci"] == ""
P.loc[kosong, "kunci"] = ["_kosong_" + str(i) for i in P.index[kosong]]   # emoji-only: tiap baris sendiri
grup = P["kunci"].astype("category").cat.codes

# ---------- bagi ulang ----------
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM)
idx_train, idx_test = next(sgkf.split(P, P["label"], groups=grup))
train, test = P.iloc[idx_train].copy(), P.iloc[idx_test].copy()

assert set(train["kunci"]).isdisjoint(set(test["kunci"])), "Masih ada teks identik di train & uji!"

# ---------- simpan (skema kolom sama dengan yang dibaca skrip model) ----------
tr_out = train.rename(columns={"label": "label_final"})
tr_out["ada_di_uji"] = False
tr_out[["rs", "rating", "teks", "label_final", "perlu_cek", "ada_di_uji", "asal", "catatan"]] \
    .to_csv("train_v3.csv", index=False, encoding="utf-8-sig")

te_out = test.rename(columns={"rs": "rs_name", "teks": "text"})
te_out[["rs_name", "rating", "text", "label", "perlu_cek", "asal", "catatan"]] \
    .to_csv("test_v3.csv", index=False, encoding="utf-8-sig")

# ---------- ringkasan ----------
print(f"train_v3: {len(train)} {train['label'].value_counts().to_dict()}")
print(f"test_v3 : {len(test)} {test['label'].value_counts().to_dict()}")
print("asal baris uji baru :", test["asal"].value_counts().to_dict())
print("asal baris train baru:", train["asal"].value_counts().to_dict())