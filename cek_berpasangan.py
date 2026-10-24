"""
cek_berpasangan.py
Evaluasi berpasangan: svm_pipeline_final.pkl vs svm_pipeline.pkl
pada test set berlabel manusia dan sampel acak terbaru.

Catatan penting:
- test_set_clean.csv memakai kolom 'clean_text' (sudah dipraproses)
- sampel_label_terisi.csv memakai kolom 'Isi Ulasan' (teks mentah)
- Jika pipeline TIDAK punya cleaning internal, aktifkan CLEAN_SAMPEL=True
  dan sediakan fungsi clean_text() yang sama dengan yang dipakai
  saat membuat 'clean_text'.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

# ---------------------------------------------------------------- konstanta
L = ["Negatif", "Netral", "Positif"]
RNG = np.random.default_rng(42)

MODEL_FINAL = "svm_pipeline_final.pkl"
MODEL_LAMA = "svm_pipeline.pkl"

CAND_TEKS = ["clean_text", "Isi Ulasan", "review", "text", "ulasan", "ulasan_bersih", "content"]
CAND_LABEL = ["label_manual", "label", "sentimen", "target"]

# Set True jika pipeline TIDAK punya step cleaning internal
# dan kolom 'Isi Ulasan' pada sampel perlu dibersihkan dulu.
CLEAN_SAMPEL = False

# Isi dengan fungsi cleaning Anda, mis.:
# from utils import clean_text
clean_text_fn = None  # ganti: clean_text_fn = clean_text


# ---------------------------------------------------------------- load model
def load(path):
    """Kembalikan (pipeline, label_encoder|None)."""
    o = joblib.load(path)
    if isinstance(o, dict):
        return o["pipeline"], o.get("label_encoder")
    return o, None


def predict(path, X):
    """Prediksi string label. Tangani 3 skenario: encoder, int, string."""
    pipe, le = load(path)
    p = np.asarray(pipe.predict(X))
    if le is not None:
        return np.asarray(le.inverse_transform(p))
    if np.issubdtype(p.dtype, np.integer):
        return np.asarray([L[i] for i in p])
    return p


# ---------------------------------------------------------------- metrik
def macro(y, p):
    return f1_score(y, p, average="macro", labels=L, zero_division=0)


# ---------------------------------------------------------------- deteksi kolom
def _detect(cols, candidates, kind, csv):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Kolom {kind} tidak ditemukan di {csv}. Kolom tersedia: {cols}")


# ---------------------------------------------------------------- evaluasi
def evaluasi(nama, csv, teks=None, label=None, do_clean=None):
    df = pd.read_csv(csv, encoding="utf-8-sig")
    cols = df.columns.tolist()

    if teks is None:
        teks = _detect(cols, CAND_TEKS, "teks", csv)
    if label is None:
        label = _detect(cols, CAND_LABEL, "label", csv)

    df = df.dropna(subset=[teks, label])
    X = df[teks].astype(str).to_numpy()
    y = df[label].astype(str).str.strip().str.capitalize().to_numpy()

    # opsi cleaning untuk menyamakan domain input
    if do_clean is None:
        do_clean = CLEAN_SAMPEL and (teks != "clean_text")
    if do_clean:
        if clean_text_fn is None:
            raise RuntimeError(
                "CLEAN_SAMPEL=True tapi clean_text_fn belum di-set. "
                "Import fungsi cleaning Anda dan set clean_text_fn = clean_text."
            )
        print(f"  [clean] menerapkan cleaning pada kolom '{teks}'")
        X = np.array([clean_text_fn(t) for t in X], dtype=object)

    # buang label di luar L (typo dll.)
    mask = np.isin(y, L)
    if (~mask).any():
        bad = sorted(set(y[~mask].tolist()))
        print(f"⚠️  {nama}: buang {int((~mask).sum())} baris berlabel di luar {L}: {bad}")
        X, y = X[mask], y[mask]

    if len(y) == 0:
        print(f"⚠️  {nama}: tidak ada baris valid, dilewati.")
        return

    # prediksi kedua model
    pa = predict(MODEL_FINAL, X)
    pb = predict(MODEL_LAMA, X)

    # metrik
    fa, fb = macro(y, pa), macro(y, pb)
    pr = precision_recall_fscore_support(y, pa, labels=L, zero_division=0)

    # ------------------------------------------------ laporan
    print(f"\n== {nama} (n={len(y)}) ==")
    print(f"  kolom teks='{teks}', label='{label}'")
    print(f"  distribusi label : {pd.Series(y).value_counts().to_dict()}")
    print(f"  distribusi final : {pd.Series(pa).value_counts().to_dict()}")
    print(f"  distribusi lama  : {pd.Series(pb).value_counts().to_dict()}")
    print("  -- performa model final per kelas --")
    for i, c in enumerate(L):
        print(f"    {c:8s}  P={pr[0][i]:.3f}  R={pr[1][i]:.3f}  "
              f"F1={pr[2][i]:.3f}  n={pr[3][i]}")

    # ------------------------------------------------ bootstrap berpasangan
    d = np.empty(2000)
    n = len(y)
    for k in range(2000):
        idx = RNG.integers(0, n, n)
        d[k] = macro(y[idx], pa[idx]) - macro(y[idx], pb[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])

    print(f"  F1-macro final {fa:.4f} | lama {fb:.4f} | selisih {fa - fb:+.4f}")
    sig = ("→ SIGNIFIKAN (0 di luar CI)"
           if (lo > 0 or hi < 0) else "→ TIDAK signifikan (0 dalam CI)")
    print(f"  CI 95% selisih (bootstrap berpasangan, 2000×): [{lo:.4f}, {hi:.4f}]  {sig}")

    # peringatan kelas minoritas
    min_n = int(pd.Series(y).value_counts().min())
    if min_n < 20:
        print(f"  ⚠️  kelas minoritas hanya n={min_n} → F1-macro & CI tidak stabil, "
              f"jangan klaim perbedaan tanpa hati-hati.")


# ---------------------------------------------------------------- sanity check
def cek_kolom_dan_distribusi(paths):
    print("=== sanity check kolom & distribusi label ===")
    for f, col_hint in paths:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", nrows=0)
            cols = df.columns.tolist()
            col = col_hint if col_hint in cols else _detect(cols, CAND_LABEL, "label", f)
            s = (pd.read_csv(f, encoding="utf-8-sig")[col]
                 .dropna().astype(str).str.strip().str.capitalize())
            print(f"  {f}")
            print(f"    kolom : {cols}")
            print(f"    label ('{col}') : {s.value_counts().to_dict()}")
        except Exception as e:
            print(f"  {f} → ERROR: {e}")


def cek_leakage(file_train_list, file_sampel, teks=None):
    """Cek irisan teks antara file sampel dan file-file training."""
    print("\n=== cek overlap teks (potensi leakage) ===")
    samp = pd.read_csv(file_sampel, encoding="utf-8-sig")
    if teks is None:
        teks = _detect(samp.columns.tolist(), CAND_TEKS, "teks", file_sampel)
    set_samp = set(samp[teks].dropna().astype(str))
    for f in file_train_list:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            tc = teks if teks in df.columns else _detect(df.columns.tolist(), CAND_TEKS, "teks", f)
            overlap = set_samp & set(df[tc].dropna().astype(str))
            print(f"  {file_sampel} ∩ {f} : {len(overlap)} teks beririsan")
        except FileNotFoundError:
            print(f"  {f} → tidak ditemukan, dilewati")


def cek_isi_pipeline(path=MODEL_FINAL):
    """Tampilkan struktur pipeline untuk cek apakah ada cleaning internal."""
    print(f"\n=== struktur {path} ===")
    pipe, le = load(path)
    print(f"  {pipe}")
    if le is not None:
        print(f"  label_encoder.classes_ = {list(le.classes_)}")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    TEST = "data/training/test_set_clean.csv"
    SAMPEL = "sampel_label_terisi.csv"

    # 0) lihat struktur pipeline (cek ada cleaning internal atau tidak)
    cek_isi_pipeline(MODEL_FINAL)

    # 1) sanity check kolom & distribusi label
    cek_kolom_dan_distribusi([(TEST, "label"), (SAMPEL, "label_manual")])

    # 2) evaluasi (auto-detect kolom teks & label)
    evaluasi("Test set berlabel manusia", TEST)
    evaluasi("Sampel acak terbaru", SAMPEL)

    # 3) cek leakage — sesuaikan path file training Anda
    cek_leakage(
        file_train_list=["data/training/train_set_clean.csv"],  # ganti jika nama berbeda
        file_sampel=SAMPEL,
    )