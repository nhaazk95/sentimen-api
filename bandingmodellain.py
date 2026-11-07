# bandingkan_sampling_model_lain.py
# Membandingkan 4 strategi penanganan ketidakseimbangan kelas untuk model SELAIN SVM
# (SVM sudah ada di bandingkan_imbalance.py / cv_berulang_imbalance.py):
#     Naive Bayes, Complement NB, Logistic Regression, Random Forest
# Strategi: Tanpa penanganan | Class weight | Random undersampling | Random oversampling
#   - "Class weight" untuk Naive Bayes = prior kelas seragam (fit_prior=False), karena NB tidak punya
#     parameter class_weight. Untuk Complement NB strategi ini TIDAK BERLAKU (prior tidak dipakai saat
#     prediksi, hasilnya identik dengan "tanpa penanganan"), sehingga dilewati.
#   - Sampling berada di dalam imblearn Pipeline -> hanya terjadi di lipatan latih (tanpa kebocoran).
#   - Parameter dipilih lewat CV 5-lipatan di DATA LATIH (lipatan anti-duplikat);
#     data uji dipakai SEKALI untuk pelaporan, dengan CI 95% bootstrap.
#
# Instalasi: pip install imbalanced-learn
# Jalankan:  python bandingkan_sampling_model_lain.py
# Opsi:      ONLY="Random Forest"   (jalankan satu model saja; nama persis seperti di MODELS)
#            QUICK=1                (uji cepat, data diperkecil; angka TIDAK untuk dilaporkan)
#            CHAR=1                 (tambah TF-IDF karakter, seperti di skrip bandingkan_model Anda)
#            PREPROC=kustom         (pakai aturan negasi kustom; default: umum)
import os, re, time, warnings, unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
warnings.filterwarnings("ignore")

TRAIN_FILE, TEST_FILE = "train_fixed.csv", "test2_fixed.csv"
RANDOM = 42
LABELS = ["negatif", "netral", "positif"]
QUICK  = os.environ.get("QUICK") == "1"
CHAR   = os.environ.get("CHAR") == "1"
PREPROC = os.environ.get("PREPROC", "umum")
N_BOOT = 1000

# ---------------- pembersihan teks ----------------
EMO_POS = r"[👍👌🌟⭐✨❤♥💗💖💓💙🩷🤩😍🥰🫶🔥💯]"; EMO_NEG = r"[👎😡🤬]"
NEGW = r"(?:tidak|ga|gak|nggak|ngga|tdk|tak)"
NEG_UMUM = re.compile(r"\b" + NEGW + r"\s+(?:ada\s+)?(?:kekurangan|kurangnya|kurang|mengecewakan|kecewa|jutek|judes|lama|lelet|ribet)\b")
# Aturan kustom (dari skrip Anda). Catatan: dibuat setelah melihat contoh di data uji; pada CV data latih
# tidak memberi peningkatan berarti, sehingga sebaiknya dilaporkan terbuka bila dipakai.
RULES_KUSTOM = [
    (NEGW + r"\s+(?:kelamaan|lama)", " cepat "), (NEGW + r"\s+(?:lambat|lelet)", " cepat "),
    (NEGW + r"\s+siang", " pagi "), (NEGW + r"\s+membedabedakan", " adil "),
    (NEGW + r"\s+(?:ada\s+)?(?:kekurangan|kurangnya|kurang|masalah)", " sempurna "),
    (NEGW + r"\s+(?:jauh|lama|lelet|mahal|ribet|kecewa|mengecewakan)", " bagus "),
    (NEGW + r"\s+(?:usah|perlu)\s+bawa", " bagus "), (NEGW + r"\s+bakal\s+lupain?", " suka "),
    (r"(?:memeuaskann|memeuaskan|muas|muaskn|memuaskan)", " puas "), (r"(?:bgs|bgus|bagus|mantap|mantul|keren)", " bagus "),
]
def bersihkan(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\\[rn]|<br>", " ", s); s = re.sub(r"http\S+", " ", s)
    if PREPROC == "kustom":
        for pat, rep in RULES_KUSTOM: s = re.sub(r"\b" + pat + r"\b", rep, s)
    s = re.sub(EMO_POS, " emopos ", s); s = re.sub(EMO_NEG, " emoneg ", s)
    s = re.sub(r"[^a-z\s]", " ", s); s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NEG_UMUM.sub(" tanpakeluhan ", s) if PREPROC == "umum" else s

def kunci_grup(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\\[rn]|<br>|[\r\n]+", " ", s))).strip()

# ---------------- data ----------------
train = pd.read_csv(TRAIN_FILE); test = pd.read_csv(TEST_FILE)
train["y"] = train["label_final"].astype(str).str.lower().str.strip()
train = train[train["y"].isin(LABELS)]
if "ada_di_uji" in train: train = train[~train["ada_di_uji"].astype(str).str.lower().eq("true")]
if "perlu_cek" in train:  train = train[train["perlu_cek"].isna()]
test["y"] = test["label"].astype(str).str.lower().str.strip()
test = test[~test["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)]
train["x"] = train["teks"].map(bersihkan); test["x"] = test["text"].map(bersihkan)
train = train[train["x"] != ""]
if QUICK:
    train = pd.concat([g.sample(min(len(g), 600), random_state=RANDOM) for _, g in train.groupby("y")])
train = train.reset_index(drop=True)
X, y = train["x"], train["y"]
grup = pd.Series(train["teks"].map(kunci_grup)).astype("category").cat.codes.values
TUNE_SPLITS = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM).split(X, y, groups=grup))
print(f"[INFO] train: {len(train)} {y.value_counts().to_dict()} | test: {len(test)} {test['y'].value_counts().to_dict()}")
print(f"[INFO] PREPROC={PREPROC} | CHAR={CHAR} | QUICK={QUICK}")

def tfidf():
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=20000)
    if not CHAR: return w
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=30000)
    return FeatureUnion([("word", w), ("char", c)])

# ---------------- model, grid, dan cara "class weight" per model ----------------
# fungsi pembuat: (apakah_class_weight) -> estimator
MODELS = {
    "Naive Bayes": dict(
        mk=lambda cw: MultinomialNB(fit_prior=not cw),
        grid={"clf__alpha": [0.1, 0.5, 1.0]}),
    "Complement NB": dict(
        mk=lambda cw: ComplementNB(fit_prior=not cw),
        grid={"clf__alpha": [0.1, 0.5, 1.0]}),
    "Logistic Regression": dict(
        mk=lambda cw: LogisticRegression(max_iter=3000, class_weight="balanced" if cw else None, random_state=RANDOM),
        grid={"clf__C": [1, 10, 100]}),
    "Random Forest": dict(   # parameter hasil penyetelan lewat CV di data latih (max_features 0.05 terbaik)
        mk=lambda cw: RandomForestClassifier(n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=RANDOM,
                                             class_weight="balanced_subsample" if cw else None),
        grid={"clf__max_features": [0.05, "sqrt"]}),
}
if QUICK:
    MODELS["Random Forest"]["grid"] = {"clf__max_features": [0.05]}
    for k in ("Logistic Regression",): MODELS[k]["grid"] = {"clf__C": [10]}
if os.environ.get("ONLY"):
    MODELS = {k: v for k, v in MODELS.items() if k == os.environ["ONLY"]}
    assert MODELS, "Nama model tidak dikenal. Pilih: Naive Bayes | Complement NB | Logistic Regression | Random Forest"

STRATEGI = {   # (sampler, class_weight?)
    "Tanpa penanganan": (None, False),
    "Class weight":     (None, True),
    "Undersampling":    (lambda: RandomUnderSampler(random_state=RANDOM), False),
    "Oversampling":     (lambda: RandomOverSampler(random_state=RANDOM), False),
}

def bootstrap_ci(yt, yp, fn, seed=RANDOM):
    rng = np.random.default_rng(seed); yt, yp = np.asarray(yt), np.asarray(yp); v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yt), len(yt)); v.append(fn(yt[i], yp[i]))
    return np.percentile(v, [2.5, 97.5])
f1m  = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
recn = lambda a, b: recall_score(a, b, labels=["negatif"], average=None, zero_division=0)[0]

# ---------------- latih & evaluasi ----------------
rows = []
for mnama, m in MODELS.items():
    for snama, (smp, cw) in STRATEGI.items():
        if mnama == "Complement NB" and snama == "Class weight":
            print(f"\n=== {mnama} | {snama} === dilewati (tidak berlaku: prior tidak dipakai Complement NB)")
            continue
        print(f"\n=== {mnama} | {snama} ===", flush=True)
        steps = [("tfidf", tfidf())]
        if smp is not None: steps.append(("sampler", smp()))
        steps.append(("clf", m["mk"](cw)))
        t0 = time.time()
        gs = GridSearchCV(Pipeline(steps), m["grid"], cv=TUNE_SPLITS, scoring="f1_macro", n_jobs=1).fit(X, y)
        waktu = time.time() - t0
        pr = gs.predict(test["x"]); yt = test["y"]
        rec = recall_score(yt, pr, labels=LABELS, average=None, zero_division=0)
        prc = precision_score(yt, pr, labels=LABELS, average=None, zero_division=0)
        cf, cr = bootstrap_ci(yt, pr, f1m), bootstrap_ci(yt, pr, recn)
        print(f"best params   : {gs.best_params_}  (F1-macro CV: {gs.best_score_:.4f}; {waktu:.0f}s)")
        print(f"F1-macro uji  : {f1m(yt, pr):.4f}  CI95 [{cf[0]:.3f}, {cf[1]:.3f}] | Accuracy {accuracy_score(yt, pr):.4f}")
        print(f"Recall negatif: {rec[0]:.4f}  CI95 [{cr[0]:.3f}, {cr[1]:.3f}] | Recall neg/net/pos {rec[0]:.3f}/{rec[1]:.3f}/{rec[2]:.3f}")
        rows.append({"Model": mnama, "Strategi": snama, "Best Params": str(gs.best_params_),
                     "F1-macro CV": round(gs.best_score_, 4), "F1-macro Uji": round(f1m(yt, pr), 4),
                     "CI95 F1": f"[{cf[0]:.3f}, {cf[1]:.3f}]", "Recall Negatif": round(rec[0], 4),
                     "CI95 RecNeg": f"[{cr[0]:.3f}, {cr[1]:.3f}]", "Recall Netral": round(rec[1], 4),
                     "Precision Negatif": round(prc[0], 4), "Accuracy": round(accuracy_score(yt, pr), 4),
                     "Waktu (s)": round(waktu, 1)})

res = pd.DataFrame(rows)
suf = (("_" + os.environ["ONLY"].replace(" ", "")) if os.environ.get("ONLY") else "") + ("_quick" if QUICK else "")
res.to_csv(f"hasil_sampling_model_lain{suf}.csv", index=False)
print("\n\n===== RINGKASAN =====")
print(res[["Model", "Strategi", "F1-macro CV", "F1-macro Uji", "Recall Negatif", "Recall Netral", "Accuracy"]].to_string(index=False))
for mnama in res["Model"].unique():
    d = res[res["Model"] == mnama]
    print(f"\n{mnama}: F1-macro uji tertinggi = {d.loc[d['F1-macro Uji'].idxmax(), 'Strategi']} | "
          f"recall negatif tertinggi = {d.loc[d['Recall Negatif'].idxmax(), 'Strategi']}")
print("\nCatatan: bila CI 95% antar strategi saling tumpang tindih, perbedaannya belum dapat disimpulkan nyata.")
print(f"Tersimpan: hasil_sampling_model_lain{suf}.csv")