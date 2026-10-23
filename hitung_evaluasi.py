"""
hitung_evaluasi.py — versi diperbaiki
Evaluasi independen model sentimen pada sampel berlabel manual.

Perbaikan:
1. Guard jika label_manual kosong -> pesan error jelas (bukan shape=(0,5000)).
2. Mapping label fleksibel (Positif/positif/POSITIF/1/2/3/pos/neg/net).
3. Opsi fallback label dari kolom Rating (default False).
4. Cek kolom wajib sebelum preprocessing.
5. Encoding CSV aman (utf-8-sig untuk file dari Excel).
"""

import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from preprocessing import full_preprocess, load_slang_dict

LABELS = ["Negatif", "Netral", "Positif"]
FILE = "sampel_label_terisi.csv"   # <-- pakai file HASIL pelabelan, bukan sampel_label.csv
PIPE_PATH = "svm_pipeline.pkl"
ENCODER_PATH = "label_encoder.pkl"
SLANG_PATH = "slang_dict.json"

# Mapping fleksibel: apapun isi CSV, dinormalisasi ke 3 label valid
LABEL_MAP = {
    "positif": "Positif", "positive": "Positif", "pos": "Positif",
    "3": "Positif", "puas": "Positif", "bagus": "Positif",
    "negatif": "Negatif", "negative": "Negatif", "neg": "Negatif",
    "1": "Negatif", "kecewa": "Negatif", "buruk": "Negatif",
    "netral": "Netral", "neutral": "Netral", "net": "Netral",
    "2": "Netral", "biasa": "Netral",
}

# Jika True, baris tanpa label_manual akan diisi dari Rating (1-5).
# PERINGATAN: ini BUKAN label manual independen -> bias ke rating.
FALLBACK_FROM_RATING = False


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    # Cek kolom wajib
    if "Isi Ulasan" not in df.columns:
        raise SystemExit(
            f"Kolom 'Isi Ulasan' tidak ditemukan.\n"
            f"Kolom yang ada: {df.columns.tolist()}"
        )

    # Normalisasi label_manual
    if "label_manual" not in df.columns:
        df["label_manual"] = np.nan

    df["label_manual"] = (
        df["label_manual"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_MAP)          # NaN jika tidak dikenali
    )

    # Fallback opsional dari Rating
    if FALLBACK_FROM_RATING and "Rating" in df.columns:
        mask = df["label_manual"].isna()
        rating = pd.to_numeric(df.loc[mask, "Rating"], errors="coerce")
        df.loc[mask, "label_manual"] = np.select(
            [rating >= 4, rating <= 2, rating == 3],
            ["Positif", "Negatif", "Netral"],
            default=np.nan,
        )
        print(f"[warn] {mask.sum()} baris dilabeli dari Rating (BUKAN label manual).")

    n_before = len(df)
    df = df[df["label_manual"].isin(LABELS)].reset_index(drop=True)
    n_after = len(df)

    if n_after == 0:
        raise SystemExit(
            f"Tidak ada baris valid setelah filter label_manual.\n"
            f"Total baris dibaca: {n_before}\n"
            f"Kolom label_manual kemungkinan belum diisi di {path}.\n"
            f"Isi dengan salah satu: {LABELS} (case-insensitive).\n"
            f"Atau jalankan label_manual_helper_fast.py dulu untuk mengisi.\n"
            f"Atau set FALLBACK_FROM_RATING=True untuk smoke-test (tidak direkomendasikan)."
        )

    if n_after < n_before:
        print(f"[info] {n_before - n_after} baris dibuang karena label tidak valid.")

    return df


def wilson(k: int, n: int, z: float = 1.96):
    """Interval kepercayaan 95% Wilson untuk proporsi k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    if n == 0:
        return "tidak ada data"
    return f"{k}/{n} = {k / n:.4f}  (95% CI {lo:.4f} - {hi:.4f})"


def main():
    # Muat artefak model
    try:
        pipe = joblib.load(PIPE_PATH)
        le = joblib.load(ENCODER_PATH)
    except FileNotFoundError as e:
        raise SystemExit(f"Artefak model tidak ditemukan: {e}")
    slang = load_slang_dict(SLANG_PATH)

    # Muat & validasi data
    df = load_data(FILE)
    n = len(df)

    # Preprocessing teks
    X = df["Isi Ulasan"].astype(str).apply(lambda t: full_preprocess(t, slang))

    if len(X) == 0 or X.isna().all():
        raise SystemExit("X kosong setelah preprocessing. Cek isi kolom 'Isi Ulasan'.")

    y_true = df["label_manual"].values
    y_pred = le.inverse_transform(pipe.predict(X))

    print("jumlah ulasan berlabel:", n)
    print("sebaran label manual:", df["label_manual"].value_counts().to_dict())
    print("\naccuracy:", fmt(int((y_true == y_pred).sum()), n))

    print("\nper kelas:")
    for lab in LABELS:
        tp = int(((y_true == lab) & (y_pred == lab)).sum())
        n_true = int((y_true == lab).sum())
        n_pred = int((y_pred == lab).sum())
        print(f"  {lab}")
        print("    recall   :", fmt(tp, n_true))
        print("    precision:", fmt(tp, n_pred))

    # Bootstrap F1-macro
    rng = np.random.default_rng(42)
    boot = [
        f1_score(y_true[s], y_pred[s], labels=LABELS, average="macro", zero_division=0)
        for s in (rng.choice(n, size=n, replace=True) for _ in range(2000))
    ]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    f1m = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    print(f"\nF1-macro: {f1m:.4f}  (95% CI bootstrap {lo:.4f} - {hi:.4f})")

    print("\n", classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0))
    print("confusion matrix (baris=manual, kolom=prediksi), urutan:", LABELS)
    print(confusion_matrix(y_true, y_pred, labels=LABELS))


if __name__ == "__main__":
    sys.exit(main())