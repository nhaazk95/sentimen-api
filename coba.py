import pandas as pd
import joblib

# 1. Load data & pipeline
df   = pd.read_csv("data_gabungan_final.csv")
pipe = joblib.load("svm_pipeline_final.pkl")

# 2. Prediksi (pipeline terima teks mentah langsung)
df["sentimen_kode"] = pipe.predict(df["clean"].fillna(""))

# 3. Cek distribusi kode dulu — JANGAN langsung mapping
print("Distribusi kode prediksi:")
print(df["sentimen_kode"].value_counts().sort_index())
print()

# 4. Cek beberapa contoh per kode untuk tahu mana negatif/positif
for kode in sorted(df["sentimen_kode"].unique()):
    contoh = df[df["sentimen_kode"] == kode].head(3)
    print(f"--- Kode {kode} ---")
    for _, r in contoh.iterrows():
        print(f"  rating={r['rating']}  teks={r['teks'][:80]!r}")
    print()