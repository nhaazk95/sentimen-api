# laporan_akurasi.py
# Menghitung akurasi (dan metrik lain) model_terbaik1.pkl pada:
#   1) DATA LATIH  - data yang sudah dilihat model (cenderung tinggi; hanya untuk melihat overfitting)
#   2) VALIDASI CV - 5-lipatan pada data latih (model dilatih ulang tiap lipatan; opsional, ~beberapa menit)
#   3) DATA UJI    - data yang belum pernah dilihat model (INI yang dilaporkan sebagai kinerja akhir)
#
# Jalankan di folder yang berisi model_terbaik1.pkl, model_baru.py, train_fixed.csv, test2_fixed.csv:
#     python laporan_akurasi.py          (latih + uji, cepat)
#     python laporan_akurasi.py --cv     (tambah validasi CV 5-lipatan)
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
from model_baru import bersihkan          # fungsi pembersihan yang sama dengan saat pelatihan

LABELS = ["negatif", "netral", "positif"]
model = joblib.load("model_terbaik1.pkl")

# ---------- data (filter SAMA dengan bandingkanmodel.py) ----------
train = pd.read_csv("train_fixed.csv")
train["y"] = train["label_final"].astype(str).str.lower().str.strip()
train = train[train["y"].isin(LABELS)]
if "ada_di_uji" in train:
    train = train[~train["ada_di_uji"].astype(str).str.lower().eq("true")]
if "perlu_cek" in train:
    train = train[train["perlu_cek"].isna()]
train["x"] = train["teks"].map(bersihkan)
train = train[train["x"] != ""].reset_index(drop=True)

test = pd.read_csv("test2_fixed.csv")
test["y"] = test["label"].astype(str).str.lower().str.strip()
test = test[~test["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)].reset_index(drop=True)
test["x"] = test["text"].map(bersihkan)

print(f"Data latih: {len(train)} ulasan {train['y'].value_counts().to_dict()}")
print(f"Data uji  : {len(test)} ulasan {test['y'].value_counts().to_dict()}\n")

def metrik(y, p):
    rec = recall_score(y, p, labels=LABELS, average=None, zero_division=0)
    return {"Accuracy": accuracy_score(y, p),
            "Precision macro": precision_score(y, p, average="macro", zero_division=0),
            "Recall macro": recall_score(y, p, average="macro", zero_division=0),
            "F1-macro": f1_score(y, p, average="macro", zero_division=0),
            "Recall negatif": rec[0], "Recall netral": rec[1], "Recall positif": rec[2]}

hasil = {}
hasil["Data latih (sudah dilihat model)"] = metrik(train["y"], model.predict(train["x"]))

if "--cv" in sys.argv:
    print("Menjalankan CV 5-lipatan (model dilatih ulang pada tiap lipatan) ...", flush=True)
    grup = pd.Series(train["teks"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()) \
             .astype("category").cat.codes.values
    kopi = clone(model).set_params(clf__probability=False)   # prediksi sama, tapi jauh lebih cepat
    pred_cv = np.empty(len(train), dtype=object)
    for tr_i, va_i in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(train["x"], train["y"], groups=grup):
        pred_cv[va_i] = clone(kopi).fit(train["x"].iloc[tr_i], train["y"].iloc[tr_i]).predict(train["x"].iloc[va_i])
    hasil["Validasi CV 5-lipatan (data latih)"] = metrik(train["y"], pred_cv)

p_uji = model.predict(test["x"])
hasil["Data uji (belum dilihat model)"] = metrik(test["y"], p_uji)

tabel = pd.DataFrame(hasil).T.round(4)
print(tabel.to_string())
tabel.to_csv("hasil_akurasi.csv")

a_latih = tabel.iloc[0]["Accuracy"]; a_uji = tabel.iloc[-1]["Accuracy"]
f_latih = tabel.iloc[0]["F1-macro"]; f_uji = tabel.iloc[-1]["F1-macro"]
print(f"\nSelisih F1-macro latih - uji = {f_latih - f_uji:+.4f}")
print("Catatan: akurasi data latih bukan ukuran kinerja; laporkan data uji (dan CV bila ada).")
print("\nConfusion matrix data uji (baris = sebenarnya, kolom = prediksi; urutan negatif, netral, positif):")
print(confusion_matrix(test["y"], p_uji, labels=LABELS))
print("\nTersimpan: hasil_akurasi.csv")