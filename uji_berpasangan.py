"""
uji_berpasangan.py
Membandingkan model final (svm_pipeline_final.pkl) vs model lama (svm_pipeline.pkl)
pada 200 data uji manual yang sama.
Menggunakan Paired Bootstrap untuk melihat apakah selisih F1-macro signifikan.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from preprocessing import full_preprocess, load_slang_dict

L = ["Negatif", "Netral", "Positif"]
slang = load_slang_dict("slang_dict.json")

# 1. Muat data uji yang sama (200 baris timpang)
df = pd.read_csv("sampel_label_terisi.csv", encoding="utf-8-sig")
df["label_manual"] = df["label_manual"].astype(str).str.strip().str.capitalize()
df = df[df["label_manual"].isin(L)].reset_index(drop=True)

# Preprocess teks SEKALI saja agar adil
X_text = df["Isi Ulasan"].astype(str).apply(lambda t: full_preprocess(t, slang))
y_true = df["label_manual"].to_numpy()

# 2. Fungsi muat model & prediksi
def prediksi(path_pipe, path_le):
    pipe = joblib.load(path_pipe)
    le = joblib.load(path_le)
    return le.inverse_transform(pipe.predict(X_text))

# 3. Dapatkan prediksi dari kedua model
print("[info] Memprediksi dengan Model Final (svm_pipeline_final.pkl)...")
pa = prediksi("svm_pipeline_final.pkl", "label_encoder_final.pkl")

print("[info] Memprediksi dengan Model Lama (svm_pipeline.pkl)...")
pb = prediksi("svm_pipeline.pkl", "label_encoder.pkl")

# 4. Hitung F1-macro aktual
f1_a = f1_score(y_true, pa, average="macro", labels=L, zero_division=0)
f1_b = f1_score(y_true, pb, average="macro", labels=L, zero_division=0)
print(f"\nF1-macro Model Final : {f1_a:.4f}")
print(f"F1-macro Model Lama  : {f1_b:.4f}")
print(f"Selisih (Final - Lama) : {f1_a - f1_b:.4f}")

# 5. Paired Bootstrap (2.000 sampel)
print("\n[info] Menjalankan Paired Bootstrap (2.000 iterasi)...")
rng = np.random.default_rng(42)
d = []
for _ in range(2000):
    idx = rng.integers(0, len(y_true), len(y_true))
    # Hitung selisih F1-macro pada sampel bootstrap
    score_a = f1_score(y_true[idx], pa[idx], average="macro", labels=L, zero_division=0)
    score_b = f1_score(y_true[idx], pb[idx], average="macro", labels=L, zero_division=0)
    d.append(score_a - score_b)

d = np.array(d)
ci_low, ci_high = np.percentile(d, [2.5, 97.5])

print(f"Selisih F1-macro (Final - Lama) 95% CI: [{ci_low:.4f} - {ci_high:.4f}]")

# 6. Kesimpulan
print("\nKesimpulan:")
if ci_low > 0:
    print("✅ CI tidak melewati nol (sepenuhnya positif).")
    print("   Model FINAL SIGNIFIKAN lebih baik dari model LAMA pada data uji ini.")
    print("   -> AMAN untuk menulis 'lebih baik dari model lama' di laporan.")
elif ci_high < 0:
    print("✅ CI tidak melewati nol (sepenuhnya negatif).")
    print("   Model LAMA SIGNIFIKAN lebih baik dari model FINAL pada data uji ini.")
    print("   -> JANGAN menulis 'lebih baik dari model lama' di laporan.")
else:
    print("⚠️ CI melewati nol.")
    print("   Tidak ada bukti statistik bahwa salah satu model lebih baik.")
    print("   -> JANGAN tulis 'lebih baik dari model lama' di laporan.")