# bandingkan_model.py
import os, re, time, warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_score
warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI
# ============================================================
TRAIN_FILE = "train_fixed.csv"
TEST_FILE  = "test2_fixed.csv"

RANDOM = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)

USE_STEM          = False  # True = stemming Sastrawi (pip install Sastrawi), sama untuk train & uji
EXCLUDE_LEAK      = True   # buang dari train teks yang juga ada di data uji (ada_di_uji)
EXCLUDE_UNCERTAIN = True   # buang dari train baris campuran/ambigu (perlu_cek terisi)
QUICK             = os.environ.get("QUICK") == "1"  # uji cepat: hanya NB & LR

LABELS = ["negatif", "netral", "positif"]

# ============================================================
# PEMBERSIHAN TEKS (SATU FUNGSI UNTUK TRAIN & UJI)
# ============================================================
EMO_POS = r"[👍👌🌟⭐✨❤♥💗💖💓💙🩷🤩😍🥰🫶🔥💯]"
EMO_NEG = r"[👎😡🤬]"

_stemmer = None
if USE_STEM:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from functools import lru_cache
    _s = StemmerFactory().create_stemmer()
    @lru_cache(maxsize=None)
    def _stem_word(w): return _s.stem(w)
    _stemmer = _stem_word

def bersihkan(s):
    s = str(s).lower()
    s = re.sub(r"\\[rn]|<br>", " ", s)         # sisa "\r\n" literal & <br>
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(EMO_POS, " emopos ", s)          # emoji jadi token agar tidak hilang
    s = re.sub(EMO_NEG, " emoneg ", s)
    s = re.sub(r"[^a-z\s]", " ", s)             # buang angka, tanda baca, emoji lain
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)        # bagusssss -> baguss
    s = re.sub(r"\s+", " ", s).strip()
    if _stemmer:
        s = " ".join(_stemmer(w) for w in s.split())
    return s

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)
print(f"[INFO] mentah  -> train: {len(train)} | test: {len(test)}")

# --- filter train
train["y"] = train["label_final"].astype(str).str.lower().str.strip()
train = train[train["y"].isin(LABELS)]
if EXCLUDE_LEAK and "ada_di_uji" in train:
    train = train[~train["ada_di_uji"].astype(str).str.lower().eq("true")]
if EXCLUDE_UNCERTAIN and "perlu_cek" in train:
    train = train[train["perlu_cek"].isna()]

# --- filter test (label kolom 'label'); baris "[UJI SISTEM]" bukan ulasan asli
test["y"] = test["label"].astype(str).str.lower().str.strip()
if (~test["y"].isin(LABELS)).any():
    print("ERROR: ada label uji kosong/tidak valid:", test.loc[~test["y"].isin(LABELS), "y"].unique())
    raise SystemExit
test = test[~test["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)]
test_pasti = test["perlu_cek"].isna() if "perlu_cek" in test else pd.Series(True, index=test.index)

# --- teks: dari kolom MENTAH (train: 'teks', test: 'text'), dibersihkan dengan fungsi yang sama
train["x"] = train["teks"].map(bersihkan)
test["x"]  = test["text"].map(bersihkan)
train = train[train["x"] != ""]

X_train, y_train = train["x"], train["y"]
X_test,  y_test  = test["x"],  test["y"]

print(f"[INFO] dipakai -> train: {len(train)} | test: {len(test)} (pasti: {int(test_pasti.sum())})")
print(f"[INFO] distribusi train: {y_train.value_counts().to_dict()}")
print(f"[INFO] distribusi test : {y_test.value_counts().to_dict()}")

# ============================================================
# TF-IDF
# ============================================================
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=20000,
)

# ============================================================
# MODEL
# ============================================================
MODELS = {
    "Naive Bayes": {
        "clf": MultinomialNB(),
        "params": {"clf__alpha": [0.1, 0.5, 1.0]},
    },
    "Logistic Regression": {
        "clf": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM),
        "params": {"clf__C": [0.1, 1, 10, 100]},
    },
    "Complement NB": {
        "clf": ComplementNB(),
        "params": {"clf__alpha": [0.1, 0.5, 1.0]},
    },
    "Random Forest": {
        "clf": RandomForestClassifier(class_weight="balanced", random_state=RANDOM, n_jobs=-1),
        "params": {"clf__n_estimators": [200], "clf__max_depth": [None, 50]},
    },
    "SVM": {
        "clf": SVC(class_weight="balanced", random_state=RANDOM),
        "params": [
            {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10]},
            {"clf__kernel": ["rbf"], "clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.1, 0.01]},
        ],
    },
}
if os.environ.get("ONLY"):
    MODELS = {k: v for k, v in MODELS.items() if k == os.environ["ONLY"]}
if QUICK:
    MODELS = {k: MODELS[k] for k in ["Naive Bayes", "Logistic Regression"]}

# ============================================================
# LATIH & EVALUASI
# ============================================================
results, best_models = [], {}

for name, cfg in MODELS.items():
    print(f"\n=== {name} ===")
    pipe = Pipeline([("tfidf", tfidf), ("clf", cfg["clf"])])

    t0 = time.time()
    gs = GridSearchCV(pipe, cfg["params"], cv=CV, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    t_train = time.time() - t0
    best = gs.best_estimator_
    best_models[name] = best

    t0 = time.time()
    y_pred = best.predict(X_test)
    t_pred = (time.time() - t0) / len(X_test)

    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_pasti = f1_score(y_test[test_pasti], y_pred[test_pasti.values], average="macro", zero_division=0)
    acc      = (y_test.values == y_pred).mean()
    rec_neg, rec_net, rec_pos = recall_score(y_test, y_pred, labels=LABELS, average=None, zero_division=0)
    prec_neg, prec_net, prec_pos = precision_score(y_test, y_pred, labels=LABELS, average=None, zero_division=0)

    print(f"best params : {gs.best_params_}  (F1-macro CV: {gs.best_score_:.4f})")
    print(f"F1-macro    : {f1_macro:.4f} | tanpa ambigu: {f1_pasti:.4f} | Accuracy: {acc:.4f}")
    print(f"Recall  neg/net/pos: {rec_neg:.3f} / {rec_net:.3f} / {rec_pos:.3f}")
    print(f"Prec    neg/net/pos: {prec_neg:.3f} / {prec_net:.3f} / {prec_pos:.3f}")
    print(f"train {t_train:.1f}s | pred {t_pred*1000:.2f} ms/ulasan")

    results.append({
        "Model": name,
        "Best Params": str(gs.best_params_),
        "F1-macro": round(f1_macro, 4),
        "F1-macro (tanpa ambigu)": round(f1_pasti, 4),
        "Accuracy": round(acc, 4),
        "Recall Neg": round(rec_neg, 4), "Precision Neg": round(prec_neg, 4),
        "Recall Netral": round(rec_net, 4), "Precision Netral": round(prec_net, 4),
        "Train (s)": round(t_train, 1),
        "Pred (ms)": round(t_pred * 1000, 3),
    })

# ============================================================
# RINGKASAN
# ============================================================
res = pd.DataFrame(results).sort_values("F1-macro", ascending=False)
print("\n\n===== RINGKASAN =====")
print(res[["Model", "F1-macro", "F1-macro (tanpa ambigu)", "Accuracy",
           "Recall Neg", "Recall Netral"]].to_string(index=False))
res.to_csv("hasil_perbandingan.csv", index=False)

best_name = res.iloc[0]["Model"]
import joblib
joblib.dump(best_models[best_name], "model_terbaik_perbandingan.pkl")
print(f"\nModel terbaik ({best_name}) disimpan!")