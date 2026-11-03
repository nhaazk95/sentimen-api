import re
import pandas as pd
import joblib

# ============================================================
# 1. LOAD DATA & PIPELINE
# ============================================================
df = pd.read_csv("data_ulasan_compass.csv")
pipe = joblib.load("svm_pipeline_final.pkl")

print("Total baris Compass:", len(df))

# ============================================================
# 2. PREPROCESSING — SESUAIKAN DENGAN TRAINING!
# ============================================================
# Ini HANYA template. Ganti dengan fungsi preprocessing asli dari notebook-mu.

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)          # hapus <br>
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # hapus URL
    text = re.sub(r"@\w+|#\w+", " ", text)          # hapus mention/hashtag
    text = re.sub(r"[^a-z\s]", " ", text)           # hanya huruf & spasi
    text = re.sub(r"\s+", " ", text).strip()        # rapikan spasi
    # TAMBAHKAN: stopword removal, stemming (Sastrawi), dll sesuai training-mu
    return text

df["clean"] = df["teks"].apply(clean_text)

# Cek hasil
print("\nContoh teks asli vs clean:")
for i in range(3):
    print(f"  ASLI : {df['teks'].iloc[i][:100]!r}")
    print(f"  CLEAN: {df['clean'].iloc[i][:100]!r}\n")

# ============================================================
# 3. PREDIKSI
# ============================================================
df["sentimen_kode"] = pipe.predict(df["clean"].fillna(""))

mapping = {0: "negatif", 1: "netral", 2: "positif"}
df["sentimen_pred"] = df["sentimen_kode"].map(mapping)

# ============================================================
# 4. RINGKASAN
# ============================================================
print("=== DISTRIBUSI PREDIKSI ===")
print(df["sentimen_pred"].value_counts())
print()
print(df["sentimen_pred"].value_counts(normalize=True).round(3))

# Cek rating 4-5 yang diprediksi negatif
mask = (df["rating"] >= 4) & (df["rating"] <= 5) & (df["sentimen_kode"] == 0)
print(f"\nRating 4-5 diprediksi negatif: {mask.sum()}")

# Cek rating 1-2 (yang dipakai untuk bukti balasan)
mask12 = df["rating"].isin([1, 2])
print(f"Rating 1-2: {mask12.sum()}")

# ============================================================
# 5. SIMPAN
# ============================================================
df.to_csv("data_compass_with_pred.csv", index=False, encoding="utf-8-sig")
print("\n-> data_compass_with_pred.csv")