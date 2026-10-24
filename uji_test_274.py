"""
uji_test_274.py
Menguji model final (v1) pada test_set_clean.csv (274 baris).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from preprocessing import full_preprocess, load_slang_dict

L = ["Negatif", "Netral", "Positif"]
slang = load_slang_dict("slang_dict.json")

# 1. Muat data uji
# Cek file mana yang ada
import os
if os.path.exists("data/training/test_set_clean.csv"):
    FILE = "data/training/test_set_clean.csv"
    TEXT_COL = "clean_text"
    LABEL_COL = "label"
elif os.path.exists("data/training/test_set.csv"):
    FILE = "data/training/test_set.csv"
    TEXT_COL = "text"
    LABEL_COL = "label"
else:
    raise SystemExit("File test_set.csv tidak ditemukan di data/training/")

print(f"[info] Menggunakan file: {FILE}")

df = pd.read_csv(FILE, encoding="utf-8-sig")
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.capitalize()
df = df[df[LABEL_COL].isin(L)].reset_index(drop=True)

# Preprocess jika file belum clean
if TEXT_COL == "text":
    X_text = df[TEXT_COL].astype(str).apply(lambda t: full_preprocess(t, slang))
else:
    X_text = df[TEXT_COL].astype(str)

y_true = df[LABEL_COL].to_numpy()

# 2. Muat model
pipe = joblib.load("svm_pipeline_final.pkl")
le = joblib.load("label_encoder_final.pkl")
y_pred = le.inverse_transform(pipe.predict(X_text))

# 3. Cetak Hasil
print(f"\nJumlah baris valid: {len(df)}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=L))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=L, digits=4, zero_division=0))

f1m = f1_score(y_true, y_pred, average="macro", labels=L, zero_division=0)
print(f"Macro-F1: {f1m:.4f}")