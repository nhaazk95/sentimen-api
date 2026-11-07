# bandingkan_imbalance.py
# Membandingkan strategi penanganan ketidakseimbangan kelas pada SVM + TF-IDF:
#   1) tanpa penanganan   2) class weight   3) random undersampling   4) random oversampling
# Pemilihan parameter lewat CV di data latih; data uji hanya dipakai untuk pelaporan akhir.
#
# Instalasi: pip install imbalanced-learn
import os, re, time, warnings, unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             accuracy_score, confusion_matrix)
from imblearn.pipeline import Pipeline                       # Pipeline khusus imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler         # ganti ke SMOTE jika ingin
warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI
# ============================================================
TRAIN_FILE = "train_fixed.csv"
TEST_FILE  = "test.csv"
RANDOM     = 42
LABELS     = ["negatif", "netral", "positif"]

# Opsi: gabungkan negasi + keluhan jadi satu token ("tidak ada kekurangan" -> "tanpakeluhan")
USE_NEG_TOKEN = False   # atau jalankan dengan:  USE_NEG_TOKEN=1 python <skrip>
USE_NEG_TOKEN = USE_NEG_TOKEN or os.environ.get("USE_NEG_TOKEN") == "1"
SUFFIX = "_neg" if USE_NEG_TOKEN else ""     # agar hasil tidak saling menimpa
NEG = re.compile(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak|bukan)\s+(?:ada\s+)?"
                 r"(?:kekurangan|kurangnya|kurang|mengecewakan|kecewa|jutek|judes|lama|lelet|ribet)\b")
CV         = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)

EXCLUDE_LEAK      = True    # buang dari train teks yang juga ada di data uji
EXCLUDE_UNCERTAIN = True    # buang dari train baris campuran/ambigu (perlu_cek)
N_BOOT            = 1000    # jumlah resampling bootstrap untuk interval kepercayaan
QUICK             = os.environ.get("QUICK") == "1"   # uji cepat: grid kecil

# Grid SVM SAMA untuk semua strategi (agar perbandingan adil)
if QUICK:
    SVM_GRID = [{"svm__kernel": ["linear"], "svm__C": [10]}]
else:
    SVM_GRID = [
        {"svm__kernel": ["linear"], "svm__C": [1, 10]},
        {"svm__kernel": ["rbf"], "svm__C": [10], "svm__gamma": [0.1]},
    ]

# ============================================================
# PEMBERSIHAN TEKS (sama dengan bandingkan_model.py)
# ============================================================
EMO_POS = r"[👍👌🌟⭐✨❤♥💗💖💓💙🩷🤩😍🥰🫶🔥💯]"
EMO_NEG = r"[👎😡🤬]"

def bersihkan(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\\[rn]|<br>", " ", s)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(EMO_POS, " emopos ", s)
    s = re.sub(EMO_NEG, " emoneg ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    if USE_NEG_TOKEN:
        s = NEG.sub(" tanpakeluhan ", s)
    return s

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)

train["y"] = train["label_final"].astype(str).str.lower().str.strip()
train = train[train["y"].isin(LABELS)]
if EXCLUDE_LEAK and "ada_di_uji" in train:
    train = train[~train["ada_di_uji"].astype(str).str.lower().eq("true")]
if EXCLUDE_UNCERTAIN and "perlu_cek" in train:
    train = train[train["perlu_cek"].isna()]

test["y"] = test["label"].astype(str).str.lower().str.strip()
assert test["y"].isin(LABELS).all(), "Ada label uji tidak valid"
test = test[~test["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)]

train["x"] = train["teks"].map(bersihkan)
test["x"]  = test["text"].map(bersihkan)
train = train[train["x"] != ""]

X_train, y_train = train["x"], train["y"]
X_test,  y_test  = test["x"],  test["y"]
print(f"[INFO] USE_NEG_TOKEN={USE_NEG_TOKEN}")
print(f"[INFO] train: {len(train)} {y_train.value_counts().to_dict()}")
print(f"[INFO] test : {len(test)} {y_test.value_counts().to_dict()}")

# ============================================================
# STRATEGI PENANGANAN
# ============================================================
def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                           sublinear_tf=True, max_features=20000)

STRATEGI = {
    "Tanpa penanganan":   dict(sampler=None,                                   cw=None),
    "Class weight":       dict(sampler=None,                                   cw="balanced"),
    "Undersampling":      dict(sampler=RandomUnderSampler(random_state=RANDOM), cw=None),
    "Oversampling":       dict(sampler=RandomOverSampler(random_state=RANDOM),  cw=None),
}

def bangun(cfg):
    steps = [("tfidf", tfidf())]
    if cfg["sampler"] is not None:
        steps.append(("sampler", cfg["sampler"]))      # hanya aktif saat fit (tidak saat predict)
    steps.append(("svm", SVC(class_weight=cfg["cw"], random_state=RANDOM)))
    return Pipeline(steps)

def bootstrap_ci(y_true, y_pred, fn, n=N_BOOT, seed=RANDOM):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(fn(y_true[idx], y_pred[idx]))
    return np.percentile(vals, [2.5, 97.5])

f1m  = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
recn = lambda a, b: recall_score(a, b, labels=["negatif"], average=None, zero_division=0)[0]

# ============================================================
# LATIH & EVALUASI
# ============================================================
rows = []
for nama, cfg in STRATEGI.items():
    print(f"\n=== {nama} ===")
    t0 = time.time()
    gs = GridSearchCV(bangun(cfg), SVM_GRID, cv=CV, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    waktu = time.time() - t0
    y_pred = gs.best_estimator_.predict(X_test)

    rec = recall_score(y_test, y_pred, labels=LABELS, average=None, zero_division=0)
    prc = precision_score(y_test, y_pred, labels=LABELS, average=None, zero_division=0)
    f1  = f1m(y_test, y_pred)
    ci_f1, ci_rn = bootstrap_ci(y_test, y_pred, f1m), bootstrap_ci(y_test, y_pred, recn)

    print(f"best params   : {gs.best_params_}  (F1-macro CV: {gs.best_score_:.4f})")
    print(f"F1-macro uji  : {f1:.4f}  95% CI [{ci_f1[0]:.3f}, {ci_f1[1]:.3f}]")
    print(f"Recall negatif: {rec[0]:.4f}  95% CI [{ci_rn[0]:.3f}, {ci_rn[1]:.3f}]")
    print(f"Recall neg/net/pos: {rec[0]:.3f} / {rec[1]:.3f} / {rec[2]:.3f}")
    print(confusion_matrix(y_test, y_pred, labels=LABELS))

    rows.append({
        "Strategi": nama,
        "Best Params": str(gs.best_params_),
        "F1-macro CV": round(gs.best_score_, 4),
        "F1-macro Uji": round(f1, 4),
        "CI95 F1 (bawah)": round(ci_f1[0], 3), "CI95 F1 (atas)": round(ci_f1[1], 3),
        "Recall Negatif": round(rec[0], 4),
        "CI95 RecNeg (bawah)": round(ci_rn[0], 3), "CI95 RecNeg (atas)": round(ci_rn[1], 3),
        "Precision Negatif": round(prc[0], 4),
        "Recall Netral": round(rec[1], 4), "Recall Positif": round(rec[2], 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Waktu (s)": round(waktu, 1),
    })

# ============================================================
# RINGKASAN
# ============================================================
res = pd.DataFrame(rows)
res.to_csv(f"hasil_penanganan_kelas{SUFFIX}.csv", index=False)
print("\n\n===== RINGKASAN =====")
print(res[["Strategi", "F1-macro CV", "F1-macro Uji", "Recall Negatif",
           "Recall Netral", "Accuracy"]].to_string(index=False))
print("\nF1-macro uji tertinggi :", res.loc[res["F1-macro Uji"].idxmax(), "Strategi"])
print("Recall negatif tertinggi:", res.loc[res["Recall Negatif"].idxmax(), "Strategi"])
print("\nCatatan: bila CI 95% antar strategi saling tumpang tindih, perbedaannya belum "
      "dapat disimpulkan signifikan.")