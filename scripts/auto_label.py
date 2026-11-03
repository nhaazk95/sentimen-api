"""
Labeling ulasan baru: LABEL UTAMA diambil dari Rating bintang (ground truth),
BUKAN dari prediksi IndoBERT. IndoBERT di sini cuma dipakai sebagai
cross-check/QA — kalau prediksinya beda sama label dari rating, baris itu
paling perlu direview manual (kemungkinan rating-nya nggak nyambung sama
teksnya, atau teksnya ambigu/sarkas).

Input : data/incoming/*.csv   -> wajib punya kolom "text" dan "Rating" (1-5)
Output: data/staging/auto_labeled.csv        (semua baris + label + qc IndoBERT)
        data/staging/spot_check_sample.csv   (baris yang IndoBERT-nya beda
                                               sama label rating, + sample
                                               tambahan, kolom "reviewed_label"
                                               dikosongkan utk direview manual)
"""
import glob
import os
import re

import pandas as pd
from transformers import pipeline

import os
import sys

# preprocessing.py ada di ROOT repo (satu-satunya sumber, dipakai juga oleh main.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from preprocessing import full_preprocess, load_slang_dict

# --- KONFIGURASI --------------------------------------------------------
MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
SAMPLE_FRACTION = 0.10           # tambahan random sample (di luar mismatch) utk spot-check
MIN_SAMPLE = 20
MIN_TEXT_LENGTH = 3
INCOMING_DIR = "data/incoming"
STAGING_DIR = "data/staging"
LABEL_MAP = {"LABEL_0": "Positif", "LABEL_1": "Netral", "LABEL_2": "Negatif"}
# -------------------------------------------------------------------------


def clean_light(text: str) -> str:
    """Normalisasi ringan saja — TIDAK lowercase/stemming (itu tugas
    full_preprocess() di scripts/train_evaluate.py)."""
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_dedup(text: str) -> str:
    """Versi disederhanakan cuma buat BANDINGIN kemiripan antar baris (dedup
    near-duplicate) - bukan versi yang dipakai buat label/training."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Pola ulasan yang kelihatan template/copy-paste (nyebut rating bintang di
# dalam teksnya sendiri - organik biasanya nggak nulis gini, karena rating
# bintang itu field terpisah di Google Maps, bukan bagian dari teks ulasan).
TEMPLATE_PATTERN = re.compile(r"rating\s*:\s*[⭐★]+\s*\(\d/5\)", re.IGNORECASE)


def is_template_text(text: str) -> bool:
    return bool(TEMPLATE_PATTERN.search(text))


def rating_to_label(rating) -> str:
    if rating <= 2:
        return "Negatif"
    elif rating == 3:
        return "Netral"
    else:
        return "Positif"


def load_incoming() -> pd.DataFrame:
    files = glob.glob(os.path.join(INCOMING_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file baru di {INCOMING_DIR}/")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text"])

    if "Rating" not in df.columns:
        raise ValueError(
            "Kolom 'Rating' tidak ditemukan di data/incoming/. "
            "Label dibuat dari rating bintang, jadi kolom ini wajib ada."
        )

    df["text"] = df["text"].astype(str).str.strip().apply(clean_light)
    df = df[(df["text"] != "") & (df["text"].str.len() >= MIN_TEXT_LENGTH)]
    df = df.dropna(subset=["Rating"])

    # 1) Buang teks yang kelihatan template/copy-paste (bukan ulasan organik)
    before = len(df)
    df = df[~df["text"].apply(is_template_text)]
    print(f"[auto_label] Dibuang {before - len(df)} baris terdeteksi template/spam.")

    # 2) Dedup near-duplicate (bukan cuma exact match) - tangkep ulasan yang
    #    sama persis isinya tapi beda dikit di tanda baca/emoji/kapitalisasi
    before = len(df)
    df["_norm"] = df["text"].apply(normalize_for_dedup)
    df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"])
    print(f"[auto_label] Dibuang {before - len(df)} baris near-duplicate.")

    return df.reset_index(drop=True)


def main():
    df = load_incoming()
    print(f"[auto_label] {len(df)} ulasan baru (siap diberi label dari rating).")

    # Text processing BERAT (stemming, buang stopword, emoji, tanda baca) -
    # dijalanin di sini (Workflow 1), bukan ditunda sampai training.
    # Kolom "text" asli TETAP disimpan apa adanya buat kebutuhan baca manual
    # pas review PR; "text_processed" yang bakal dipakai model buat training.
    print("[auto_label] Text processing (stemming, stopword, dsb)...")
    slang_dict = load_slang_dict("slang_dict.json")
    df["text_processed"] = df["text"].apply(lambda t: full_preprocess(t, slang_dict))

    # --- Label utama: dari rating bintang (ground truth) ------------------
    df["label"] = df["Rating"].apply(rating_to_label)
    print("[auto_label] Distribusi label (dari rating):")
    print(df["label"].value_counts())

    # --- IndoBERT sebagai QA/cross-check, bukan penentu label -------------
    print("[auto_label] Menjalankan IndoBERT untuk cross-check...")
    clf = pipeline("text-classification", model=MODEL_NAME, truncation=True)
    results = clf(df["text"].tolist(), batch_size=16)

    df["indobert_pred"] = [LABEL_MAP.get(r["label"], r["label"]) for r in results]
    df["indobert_score"] = [round(r["score"], 4) for r in results]
    df["mismatch"] = df["label"] != df["indobert_pred"]

    n_mismatch = int(df["mismatch"].sum())
    print(f"[auto_label] {n_mismatch}/{len(df)} baris beda antara label rating vs prediksi IndoBERT.")

    os.makedirs(STAGING_DIR, exist_ok=True)
    labeled_path = os.path.join(STAGING_DIR, "auto_labeled.csv")
    df.to_csv(labeled_path, index=False)
    print(f"[auto_label] Disimpan -> {labeled_path}")

    # --- Sampling utk spot-check: SEMUA mismatch + sedikit random sample --
    n_sample = max(MIN_SAMPLE, int(len(df) * SAMPLE_FRACTION))
    n_sample = min(n_sample, len(df))

    mismatch_rows = df[df["mismatch"]]
    remaining = max(0, n_sample - len(mismatch_rows))
    rest_pool = df[~df["mismatch"]]

    if remaining > 0 and len(rest_pool) > 0:
        frac = min(1, remaining / len(rest_pool))
        random_sample = (
            rest_pool.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=42))
        )
    else:
        random_sample = rest_pool.iloc[0:0]

    sample = pd.concat([mismatch_rows, random_sample]).drop_duplicates(subset=["text"])
    sample = sample.reset_index(drop=True)
    sample["reviewed_label"] = ""  # diisi manual: kosongkan kalau label rating-nya sudah benar

    sample_path = os.path.join(STAGING_DIR, "spot_check_sample.csv")
    sample.to_csv(sample_path, index=False)
    print(f"[auto_label] {len(sample)} baris disampling utk spot-check "
          f"({len(mismatch_rows)} mismatch + {len(sample) - len(mismatch_rows)} random) -> {sample_path}")


if __name__ == "__main__":
    main()