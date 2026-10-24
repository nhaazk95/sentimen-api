import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# 1. Definisi Label
L = ["Negatif", "Netral", "Positif"]

# 2. Muat model & encoder (mendukung format dict maupun terpisah)
obj = joblib.load("svm_pipeline_final.pkl")
if isinstance(obj, dict):
    pipe = obj["pipeline"]
    le = obj.get("label_encoder")
else:
    pipe = obj
    # Jika model tidak menyimpan encoder di dalam dict, muat encoder terpisah
    le = joblib.load("label_encoder_final.pkl")

# 3. Muat data uji (Ganti nama kolom agar sesuai)
df = pd.read_csv("sampel_label_terisi.csv", encoding="utf-8-sig")

# Deteksi nama kolom secara otomatis
TEXT_COL = "Isi Ulasan" if "Isi Ulasan" in df.columns else "text"
LABEL_COL = "label_manual" if "label_manual" in df.columns else "label"

df = df.dropna(subset=[TEXT_COL, LABEL_COL])
y = df[LABEL_COL].astype(str).str.strip().str.capitalize().to_numpy()

# 4. Prediksi (Gunakan teks MENTAH, karena pipeline sudah punya TF-IDF dan preprocessing dilakukan di awal)
# Catatan: Jika model di latih dengan teks yang sudah di-preprocess, teks mentah ini akan diproses oleh vectorizer.
# Tapi untuk akurasi terbaik, kita preprocess dulu menggunakan fungsi dari repo Anda.
from preprocessing import full_preprocess, load_slang_dict
slang = load_slang_dict("slang_dict.json")

X_text = df[TEXT_COL].astype(str).apply(lambda t: full_preprocess(t, slang))

p = pipe.predict(X_text)
p = le.inverse_transform(p) if le is not None else np.asarray(p)

# 5. Output Metrik
print("Confusion Matrix (Baris=Aktual, Kolom=Prediksi):")
print(confusion_matrix(y, p, labels=L))

print("\nClassification Report:")
print(classification_report(y, p, labels=L, digits=4))

baseline = pd.Series(y).value_counts(normalize=True).max()
print(f"Baseline kelas mayoritas: {baseline:.4f} ({baseline*100:.2f}%)")

# 6. Bootstrap CI (2.000 sampel)
rng = np.random.default_rng(42)
bs = []
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    # Hitung macro F1 pada sampel bootstrap
    score = f1_score(y[idx], p[idx], average="macro", labels=L, zero_division=0)
    bs.append(score)

ci_low, ci_high = np.percentile(bs, [2.5, 97.5])
print(f"Macro-F1 CI 95%: [{ci_low:.4f} - {ci_high:.4f}]")