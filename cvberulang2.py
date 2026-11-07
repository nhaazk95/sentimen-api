# cv_berulang_imbalance.py
# Membandingkan 4 strategi penanganan ketidakseimbangan kelas pada SVM + TF-IDF
# (tanpa penanganan, class weight, random undersampling, random oversampling)
# memakai CV berulang 5x5 pada DATA LATIH dengan uji statistik berpasangan.
#   - lipatan SAMA untuk semua strategi (perbandingan berpasangan)
#   - teks identik tidak terpisah antara lipatan latih dan validasi (StratifiedGroupKFold)
#   - sampling hanya terjadi di dalam lipatan latih (imblearn Pipeline) -> tanpa kebocoran
#   - uji: t-test berpasangan terkoreksi Nadeau-Bengio + Wilcoxon, p dikoreksi Holm (6 pasangan)
#   - data uji dipakai SEKALI di akhir untuk pelaporan
#
# Instalasi: pip install imbalanced-learn scipy
# Jalankan:  python cv_berulang_imbalance.py
# Opsi:      RBF=1 (tambah kernel rbf; sangat lambat di oversampling) | NESTED=1 (tuning di dalam lipatan; jauh lebih lama) | QUICK=1 (uji cepat, data diperkecil)
#            USE_NEG_TOKEN=1 (token negasi)
import os, re, time, warnings, unicodedata
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, recall_score, accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
warnings.filterwarnings("ignore")

TRAIN_FILE, TEST_FILE = "train_fixed.csv", "test2_fixed.csv"
RANDOM   = 42
LABELS   = ["negatif", "netral", "positif"]
QUICK    = os.environ.get("QUICK") == "1"
N_SPLITS = 5
N_REPEAT = 1 if QUICK else 5
NESTED   = os.environ.get("NESTED") == "1"
USE_NEG_TOKEN = os.environ.get("USE_NEG_TOKEN") == "1"
N_BOOT   = 1000

# Grid SVM SAMA untuk semua strategi (3 konfigurasi). Default: SVM linear saja (cepat).
# Kernel rbf pada data hasil oversampling (21 ribu baris) sangat lambat; aktifkan dengan RBF=1.
RBF = os.environ.get("RBF") == "1"
if QUICK:
    SVM_GRID = [{"svm__kernel": ["linear"], "svm__C": [10]}]
elif RBF:
    SVM_GRID = [{"svm__kernel": ["linear"], "svm__C": [1, 10]},
                {"svm__kernel": ["rbf"], "svm__C": [10], "svm__gamma": [0.1]}]
else:
    SVM_GRID = [{"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]}]

# ---------------- pembersihan teks (sama dengan skrip lain) ----------------
EMO_POS = r"[👍👌🌟⭐✨❤♥💗💖💓💙🩷🤩😍🥰🫶🔥💯]"; EMO_NEG = r"[👎😡🤬]"
NEG = re.compile(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak|bukan)\s+(?:ada\s+)?"
                 r"(?:kekurangan|kurangnya|kurang|mengecewakan|kecewa|jutek|judes|lama|lelet|ribet)\b")
def bersihkan(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\\[rn]|<br>", " ", s); s = re.sub(r"http\S+", " ", s)

# --- 1. PENANGANAN NEGASI & TYPO (Diperluas) ---
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:ada\s+)?(?:kekurangan|kurangnya|kurang|masalah)\b", " sempurna ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:jauh|lama|lelet|mahal|ribet|kecewa|mengecewakan)\b", " bagus ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:usah|perlu)\s+bawa\b", " bagus ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+bakal\s+lupain?\b", " suka ", s)
    
    s = re.sub(r"\b(?:memeuaskann|memeuaskan|muas|muaskn|memuaskan)\b", " puas ", s)
    s = re.sub(r"\b(?:bgs|bgus|bagus|mantap|mantul|keren)\b", " bagus ", s)
       
    # --- TAMBAHKAN INI ---
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:ada\s+)?(?:kekurangan|kurangnya|kurang|masalah)\b", " sempurna ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:jauh|lama|lelet|mahal|ribet|kecewa|mengecewakan)\b", " bagus ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:kelamaan)\b", " cepat ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:lambat|lelet)\b", " cepat ", s)
    # ---------------------

    s = re.sub(EMO_POS, " emopos ", s); s = re.sub(EMO_NEG, " emoneg ", s)
    s = re.sub(r"[^a-z\s]", " ", s); s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NEG.sub(" tanpakeluhan ", s) if USE_NEG_TOKEN else s

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
if QUICK:   # uji cepat: perkecil data (hanya untuk memastikan kode berjalan)
    train = pd.concat([g.sample(min(len(g), 700), random_state=RANDOM) for _, g in train.groupby("y")])
train = train.reset_index(drop=True)
X, y = train["x"], train["y"]
grup = pd.Series(train["teks"].map(kunci_grup)).astype("category").cat.codes.values
print(f"[INFO] train: {len(train)} {y.value_counts().to_dict()} | RBF={RBF} | NESTED={NESTED} | QUICK={QUICK} | USE_NEG_TOKEN={USE_NEG_TOKEN}")

# ---------------- lipatan berulang (sama untuk semua strategi) ----------------
SPLITS = []
for r in range(N_REPEAT):
    SPLITS += list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM + r).split(X, y, groups=grup))
K = len(SPLITS)
rec_neg = lambda a, b: recall_score(a, b, labels=["negatif"], average=None, zero_division=0)[0]
rec_net = lambda a, b: recall_score(a, b, labels=["netral"], average=None, zero_division=0)[0]
SCORING = {"f1_macro": "f1_macro", "rec_neg": make_scorer(rec_neg), "rec_net": make_scorer(rec_net), "acc": "accuracy"}

def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=20000)

STRATEGI = {
    "Tanpa penanganan": dict(sampler=None,                                    cw=None),
    "Class weight":     dict(sampler=None,                                    cw="balanced"),
    "Undersampling":    dict(sampler=RandomUnderSampler(random_state=RANDOM), cw=None),
    "Oversampling":     dict(sampler=RandomOverSampler(random_state=RANDOM),  cw=None),
}
def bangun(cfg):
    steps = [("tfidf", tfidf())]
    if cfg["sampler"] is not None: steps.append(("sampler", cfg["sampler"]))
    steps.append(("svm", SVC(class_weight=cfg["cw"], random_state=RANDOM)))
    return Pipeline(steps)

TUNE_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)

# ---------------- CV berulang per strategi ----------------
skor, best = {}, {}
for nama, cfg in STRATEGI.items():
    t0 = time.time()
    if NESTED:
        est = GridSearchCV(bangun(cfg), SVM_GRID, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM),
                           scoring="f1_macro", n_jobs=1)
    else:
        gs = GridSearchCV(bangun(cfg), SVM_GRID, cv=TUNE_CV, scoring="f1_macro", n_jobs=-1, verbose=1).fit(X, y)
        best[nama] = gs.best_params_; est = gs.best_estimator_
        print(f"[TUNE] {nama}: {gs.best_params_}  (F1-macro CV tuning: {gs.best_score_:.4f})")
    print(f"[CV ] {nama}: mulai {K} lipatan ...", flush=True)
    cv = cross_validate(est, X, y, cv=SPLITS, scoring=SCORING, n_jobs=-1, verbose=1)
    skor[nama] = {m: cv[f"test_{m}"] for m in SCORING}
    print(f"[CV ] {nama}: {K} lipatan selesai dalam {time.time()-t0:.0f}s")

# ---------------- ringkasan ----------------
rows = [{"Strategi": n, **{f"{m} rata-rata": round(v.mean(), 4) for m, v in d.items()},
         **{f"{m} std": round(v.std(ddof=1), 4) for m, v in d.items()}} for n, d in skor.items()]
ring = pd.DataFrame(rows)
print(f"\n===== RINGKASAN (rata-rata ± std atas {K} lipatan) =====")
for r in rows:
    print(f"{r['Strategi']:17s} F1-macro {r['f1_macro rata-rata']:.4f}±{r['f1_macro std']:.4f} | "
          f"Recall neg {r['rec_neg rata-rata']:.4f}±{r['rec_neg std']:.4f} | "
          f"Recall netral {r['rec_net rata-rata']:.4f}±{r['rec_net std']:.4f} | Acc {r['acc rata-rata']:.4f}")

# ---------------- uji berpasangan + koreksi Holm ----------------
def nb_test(a, b, metrik):
    d = skor[a][metrik] - skor[b][metrik]
    n_tr, n_te = len(SPLITS[0][0]), len(SPLITS[0][1])
    var = d.var(ddof=1) * (1 / K + n_te / n_tr)          # koreksi Nadeau-Bengio
    se = np.sqrt(var) if var > 0 else 1e-12
    t = d.mean() / se; df = K - 1
    p = 2 * stats.t.sf(abs(t), df); h = stats.t.ppf(0.975, df) * se
    try: pw = stats.wilcoxon(d).pvalue
    except ValueError: pw = 1.0
    return d.mean(), d.mean() - h, d.mean() + h, p, pw

def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj

nama_s = list(STRATEGI)
pasang = [(a, b) for i, a in enumerate(nama_s) for b in nama_s[i + 1:]]
hasil = []
for metrik in ["f1_macro", "rec_neg", "rec_net"]:
    tmp = [nb_test(a, b, metrik) for a, b in pasang]
    padj = holm([t[3] for t in tmp])
    for (a, b), (sel, lo, hi, p, pw), pa in zip(pasang, tmp, padj):
        vonis = "SETARA" if pa >= 0.05 else f"{a if sel > 0 else b} lebih baik"
        hasil.append(dict(Metrik=metrik, Perbandingan=f"{a} - {b}", Selisih=round(sel, 4),
                          CI95_bawah=round(lo, 4), CI95_atas=round(hi, 4),
                          p_NadeauBengio=round(p, 4), p_Holm=round(pa, 4), p_Wilcoxon=round(pw, 4), Vonis=vonis))
uji = pd.DataFrame(hasil)
print("\n===== UJI BERPASANGAN (p_Holm dikoreksi untuk 6 pasangan per metrik) =====")
print(uji.to_string(index=False))

# ---------------- evaluasi akhir di data uji (SEKALI) ----------------
def bootstrap_ci(yt, yp, fn, seed=RANDOM):
    rng = np.random.default_rng(seed); yt, yp = np.asarray(yt), np.asarray(yp); v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yt), len(yt)); v.append(fn(yt[i], yp[i]))
    return np.percentile(v, [2.5, 97.5])
f1m = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
print("\n===== DATA UJI (sekali; model dilatih di seluruh data latih) =====")
akhir = []
Xtr_full = train["x"]
for nama, cfg in STRATEGI.items():
    pipe = bangun(cfg)
    if NESTED: pipe = GridSearchCV(pipe, SVM_GRID, cv=TUNE_CV, scoring="f1_macro", n_jobs=-1)
    else:      pipe.set_params(**best[nama])
    pr = pipe.fit(Xtr_full, y).predict(test["x"])
    cf, cr = bootstrap_ci(test["y"], pr, f1m), bootstrap_ci(test["y"], pr, rec_neg)
    akhir.append({"Strategi": nama, "F1-macro uji": round(f1m(test["y"], pr), 4),
                  "CI95 F1": f"[{cf[0]:.3f}, {cf[1]:.3f}]", "Recall negatif uji": round(rec_neg(test["y"], pr), 4),
                  "CI95 RecNeg": f"[{cr[0]:.3f}, {cr[1]:.3f}]", "Recall netral uji": round(rec_net(test["y"], pr), 4),
                  "Accuracy uji": round(accuracy_score(test["y"], pr), 4)})
akhir = pd.DataFrame(akhir); print(akhir.to_string(index=False))

# ---------------- simpan ----------------
suf = ("_nested" if NESTED else "") + ("_neg" if USE_NEG_TOKEN else "") + ("_quick" if QUICK else "")
ring.to_csv(f"cvimb_ringkasan{suf}.csv", index=False)
uji.to_csv(f"cvimb_uji_berpasangan{suf}.csv", index=False)
akhir.to_csv(f"cvimb_uji_akhir{suf}.csv", index=False)
pd.DataFrame({f"{n}|{m}": v for n, d in skor.items() for m, v in d.items()}).to_csv(f"cvimb_skor_per_lipatan{suf}.csv", index=False)
print("\nTersimpan: cvimb_ringkasan, cvimb_uji_berpasangan, cvimb_uji_akhir, cvimb_skor_per_lipatan (akhiran:", suf or "-", ")")