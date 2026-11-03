import os
from datetime import datetime

import pandas as pd

STAGING_DIR = "data/staging"
TRAINING_DIR = "data/training"
ARCHIVE_DIR = os.path.join(STAGING_DIR, "archive")


def main():
    labeled_path = os.path.join(STAGING_DIR, "auto_labeled.csv")
    sample_path = os.path.join(STAGING_DIR, "spot_check_sample.csv")
    pool_path = os.path.join(TRAINING_DIR, "train_pool.csv")

    if not os.path.exists(labeled_path):
        print("[prepare_dataset] Tidak ada data baru di staging (auto_labeled.csv "
              "tidak ditemukan) - lanjut retrain pakai train_pool.csv yang sudah ada.")
        return

    labeled = pd.read_csv(labeled_path)

    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        corrections = sample[sample["reviewed_label"].fillna("") != ""]
        if len(corrections):
            print(f"[prepare_dataset] Menerapkan {len(corrections)} koreksi manual.")
            labeled = labeled.merge(
                corrections[["text", "reviewed_label"]], on="text", how="left"
            )
            labeled["label"] = labeled["reviewed_label"].fillna(labeled["label"])
            labeled = labeled.drop(columns=["reviewed_label"])

    cols = ["text", "label"]
    if "text_processed" in labeled.columns:
        cols.append("text_processed")
    final = labeled[cols]

    os.makedirs(TRAINING_DIR, exist_ok=True)
    if os.path.exists(pool_path):
        old_pool = pd.read_csv(pool_path)
        combined = pd.concat([old_pool, final], ignore_index=True)
        combined = combined.drop_duplicates(subset=["text"], keep="last")
    else:
        combined = final

    combined.to_csv(pool_path, index=False)
    print(f"[prepare_dataset] train_pool.csv sekarang punya {len(combined)} baris.")

    # Arsipkan staging files supaya tidak double-proses di run berikutnya
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for f in [labeled_path, sample_path]:
        if os.path.exists(f):
            os.rename(f, os.path.join(ARCHIVE_DIR, f"{stamp}_{os.path.basename(f)}"))
    print("[prepare_dataset] Staging files diarsipkan.")


if __name__ == "__main__":
    main()