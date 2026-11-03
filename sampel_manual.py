"""
Script pembuktian metodologi skripsi:
1. Sampling manual ulasan prediksi negatif  -> proporsi + CI
2. Cek 288 ulasan rating 4-5 prediksi negatif -> export untuk verifikasi manual
3. Sampel 60 ulasan rating 1-2 (Jul 2024-Agu 2026) -> export untuk cek balasan manual

Jalankan:  python bukti_metodologi.py
Output  :  folder ./bukti_output/ berisi CSV + ringkasan.txt
"""

import os
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

# ============================================================
# KONFIGURASI
# ============================================================
CSV_PATH       = "data_gabungan_final.csv"
OUT_DIR        = "bukti_output"
SENTIMEN_COL   = "sentimen_pred"   # ganti kalau nama kolom beda
SEED           = 42
N_SAMPEL_NEG   = 50                # sampel manual ulasan negatif
N_SAMPEL_60    = 60                # sampel cek balasan
TGL_AWAL       = "2024-07-01"
TGL_AKHIR      = "2026-09-01"      # eksklusif

os.makedirs(OUT_DIR, exist_ok=True)
ringkasan = []

def log(s):
    print(s)
    ringkasan.append(s)

# ============================================================
# 0. LOAD & BERSIHKAN
# ============================================================
df = pd.read_csv(CSV_PATH)
log(f"[LOAD] Total baris: {len(df)}")
log(f"[LOAD] Kolom: {df.columns.tolist()}")

df["tanggal_ulasan"] = pd.to_datetime(df["tanggal_ulasan"], errors="coerce").dt.tz_localize(None)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Cek kolom sentimen
if SENTIMEN_COL not in df.columns:
    log(f"\n[WARN] Kolom '{SENTIMEN_COL}' TIDAK ADA di CSV.")
    log("       Script akan lanjut, tapi filter sentimen negatif dilewati.")
    log("       Untuk bukti 1a & 1c, tambahkan kolom sentimen_pred dulu")
    log("       (hasil prediksi SVM-mu), lalu jalankan ulang script ini.\n")
    ada_sentimen = False
else:
    df[SENTIMEN_COL] = df[SENTIMEN_COL].astype(str).str.lower().str.strip()
    ada_sentimen = True
    log(f"[OK] Kolom sentimen ditemukan. Distribusi:")
    log(df[SENTIMEN_COL].value_counts().to_string())

# ============================================================
# 1a. SAMPEL MANUAL ULASAN NEGATIF
# ============================================================
if ada_sentimen:
    log("\n=== BUKTI 1a: Sampel manual ulasan prediksi negatif ===")
    neg = df[df[SENTIMEN_COL] == "negatif"].copy()
    log(f"Total ulasan prediksi negatif (seluruh periode): {len(neg)}")

    if len(neg) >= N_SAMPEL_NEG:
        sampel_neg = neg.sample(n=N_SAMPEL_NEG, random_state=SEED)
    else:
        sampel_neg = neg.copy()
        log(f"[WARN] Negatif cuma {len(neg)}, pakai semua.")

    f1 = os.path.join(OUT_DIR, "1a_sampel_manual_negatif.csv")
    sampel_neg[["rs", "user", "rating", "tanggal_ulasan", "teks"]].assign(
        ada_keluhan=""  # isi manual 1/0
    ).to_csv(f1, index=False, encoding="utf-8-sig")
    log(f"-> {f1}  (isi kolom 'ada_keluhan' = 1/0, lalu jalankan mode hitung)")