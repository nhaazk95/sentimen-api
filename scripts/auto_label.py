"""
Auto-label ulasan baru pakai IndoBERT sentiment classifier, lalu sampling
sebagian untuk spot-check manual.

Input : data/incoming/*.csv   -> wajib punya kolom "text"
Output: data/staging/auto_labeled.csv        (semua baris + label + confidence)
        data/staging/spot_check_sample.csv   (sample utk direview manual,
                                               kolom "reviewed_label" dikosongkan)
"""
import glob
import os

import pandas as pd
from transformers import pipeline

# --- KONFIGURASI --------------------------------------------------------
# Ganti sesuai model IndoBERT sentiment yang kamu pakai di training awal.
MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
SAMPLE_FRACTION = 0.15          # 15% data disampling utk spot-check
MIN_SAMPLE = 20                 # minimal jumlah baris disampling
INCOMING_DIR = "data/incoming"
STAGING_DIR = "data/staging"
LABEL_MAP = {"LABEL_0": "Negatif", "LABEL_1": "Netral", "LABEL_2": "Positif"}
# -------------------------------------------------------------------------


def load_incoming() -> pd.DataFrame:
    files = glob.glob(os.path.join(INCOMING_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file baru di {INCOMING_DIR}/")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def main():
    df = load_incoming()
    print(f"[auto_label] {len(df)} ulasan baru ditemukan.")

    clf = pipeline("text-classification", model=MODEL_NAME, truncation=True)
    results = clf(df["text"].tolist())

    df["label"] = [LABEL_MAP.get(r["label"], r["label"]) for r in results]
    df["confidence"] = [round(r["score"], 4) for r in results]

    os.makedirs(STAGING_DIR, exist_ok=True)
    labeled_path = os.path.join(STAGING_DIR, "auto_labeled.csv")
    df.to_csv(labeled_path, index=False)
    print(f"[auto_label] Disimpan -> {labeled_path}")

    # Sampling stratified per label untuk spot-check
    n_sample = max(MIN_SAMPLE, int(len(df) * SAMPLE_FRACTION))
    n_sample = min(n_sample, len(df))
    sample = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(frac=min(1, n_sample / len(df)), random_state=42))
        .reset_index(drop=True)
    )
    sample["reviewed_label"] = ""  # kolom ini diisi manual saat review PR

    sample_path = os.path.join(STAGING_DIR, "spot_check_sample.csv")
    sample.to_csv(sample_path, index=False)
    print(f"[auto_label] {len(sample)} baris disampling -> {sample_path}")


if __name__ == "__main__":
    main()
