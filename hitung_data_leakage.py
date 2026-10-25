import pandas as pd
import joblib
from sklearn.metrics import f1_score

# ---------- Load data bersih (keduanya pakai clean_text) ----------
test  = pd.read_csv("data/training/test_set_clean.csv",  encoding="utf-8-sig")
train = pd.read_csv("data/training/train_pool_clean.csv", encoding="utf-8-sig")

print(f"Total test : {len(test)}")
print(f"Total train: {len(train)}")
print(f"Distribusi label test : {test['label'].value_counts().to_dict()}")

# ---------- Deteksi duplikat persis ----------
train_texts = set(train["clean_text"].dropna().astype(str).str.strip().str.lower())
test_norm   = test["clean_text"].astype(str).str.strip().str.lower()
mask_dup    = test_norm.isin(train_texts)

print(f"\nBaris test yang IDENTIK dengan train:")
print(f"  Jumlah    : {mask_dup.sum()}")
print(f"  Persentase: {mask_dup.mean()*100:.2f}%")

# ---------- Deteksi kemiripan tinggi (opsional, via cosine) ----------
# Ini melengkapi cek duplikat persis dengan cek "mirip".
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_train = vec.fit_transform(train["clean_text"].astype(str))
    X_test  = vec.transform(test["clean_text"].astype(str))
    sims    = cosine_similarity(X_test, X_train)
    max_sim = sims.max(axis=1)
    mask_sim = max_sim >= 0.90   # ambang kemiripan tinggi
    print(f"\nBaris test MIRIP (cosine >= 0.90) dengan train: {mask_sim.sum()} ({mask_sim.mean()*100:.2f}%)")
except Exception as e:
    print("Cek cosine gagal:", e)
    mask_sim = mask_dup

# ---------- Load model & evaluasi ----------
pipe = joblib.load("svm_pipeline.pkl")
if isinstance(pipe, dict): pipe = pipe["pipeline"]
L = ["Negatif", "Netral", "Positif"]

def pred_labels(X):
    p = pipe.predict(X)
    return [L[int(v)] for v in p] if p.dtype.kind in ('i','u') else p

def eval_set(df, name):
    y_true = df["label"].astype(str).str.strip().str.capitalize().to_numpy()
    y_pred = pred_labels(df["clean_text"].astype(str).to_numpy())
    f1 = f1_score(y_true, y_pred, average="macro", labels=L, zero_division=0)
    print(f"{name:40s} n={len(df):3d}  macro-F1={f1:.4f}")
    return f1

print("\n=== HASIL EVALUASI ===")
f1_full  = eval_set(test, "Full test set")
f1_clean = eval_set(test[~mask_dup], "Test set TANPA duplikat persis")
f1_sim   = eval_set(test[~mask_sim], "Test set TANPA duplikat + mirip (cos>=0.90)")