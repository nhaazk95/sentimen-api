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
import re

import pandas as pd
from transformers import pipeline

# --- KONFIGURASI --------------------------------------------------------
# Ganti sesuai model IndoBERT sentiment yang kamu pakai di training awal.
MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
SAMPLE_FRACTION = 0.15          # 15% data disampling utk spot-check
MIN_SAMPLE = 20                 # minimal jumlah baris disampling
MIN_TEXT_LENGTH = 3             # buang ulasan lebih pendek dari ini (karakter)
INCOMING_DIR = "data/incoming"
STAGING_DIR = "data/staging"
LABEL_MAP = {"LABEL_0": "Negatif", "LABEL_1": "Netral", "LABEL_2": "Positif"}
# -------------------------------------------------------------------------


def clean_light(text: str) -> str:
    """Normalisasi ringan saja — TIDAK lowercase/stemming (itu tugas
    full_preprocess() di scripts/train_evaluate.py), supaya teks yang
    dikirim ke IndoBERT tetap natural dan akurat dibaca modelnya."""
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)  # "bagusssssss" -> "baguuus"
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_incoming() -> pd.DataFrame:
    files = glob.glob(os.path.join(INCOMING_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file baru di {INCOMING_DIR}/")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip().apply(clean_light)
    df = df[(df["text"] != "") & (df["text"].str.len() >= MIN_TEXT_LENGTH)]
    df = df.drop_duplicates(subset=["text"])
    return df.reset_index(drop=True)


def flag_rating_mismatch(row) -> bool:
    """Rating bintang dipakai sebagai cross-check, BUKAN sebagai label utama
    (banyak ulasan yang teksnya netral/positif tapi bintangnya rendah karena
    alasan lain, atau sebaliknya) — tapi kalau selisihnya ekstrem, itu sinyal
    kuat auto-label IndoBERT kemungkinan salah, jadi wajib direview manual."""
    if "Rating" not in row or pd.isna(row["Rating"]):
        return False
    rating = row["Rating"]
    if rating <= 2 and row["label"] == "Positif":
        return True
    if rating >= 4 and row["label"] == "Negatif":
        return True
    return False


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

    # Tandai baris yang rating bintangnya kontradiksi sama hasil auto-label
    # (mis. rating 1 tapi label Positif) - ini prioritas tinggi utk direview,
    # kemungkinan besar auto-label-nya salah.
    df["rating_mismatch"] = df.apply(flag_rating_mismatch, axis=1)
    n_mismatch = df["rating_mismatch"].sum()
    print(f"[auto_label] {n_mismatch} baris rating vs label kontradiksi (wajib direview).")

    # Sampling: SEMUA baris mismatch masuk otomatis, sisanya diisi stratified
    # random sample per label sampai total mencapai target sample.
    n_sample = max(MIN_SAMPLE, int(len(df) * SAMPLE_FRACTION))
    n_sample = min(n_sample, len(df))

    mismatch_rows = df[df["rating_mismatch"]]
    remaining_needed = max(0, n_sample - len(mismatch_rows))
    rest_pool = df[~df["rating_mismatch"]]

    if remaining_needed > 0 and len(rest_pool) > 0:
        frac = min(1, remaining_needed / len(rest_pool))
        random_sample = (
            rest_pool.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=42))
        )
    else:
        random_sample = rest_pool.iloc[0:0]

    sample = pd.concat([mismatch_rows, random_sample]).drop_duplicates(subset=["text"])
    sample = sample.reset_index(drop=True)
    sample["reviewed_label"] = ""  # kolom ini diisi manual saat review PR

    sample_path = os.path.join(STAGING_DIR, "spot_check_sample.csv")
    sample.to_csv(sample_path, index=False)
    print(f"[auto_label] {len(sample)} baris disampling -> {sample_path}")


if __name__ == "__main__":
    main()