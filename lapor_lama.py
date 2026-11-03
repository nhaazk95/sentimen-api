import joblib, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score

L = ["Negatif", "Netral", "Positif"]

# ------------------------------------------------ load + deteksi mapping
def load_model(path):
    o = joblib.load(path)
    if isinstance(o, dict):
        pipe = o["pipeline"]
        le = o.get("label_encoder")
    else:
        pipe = o
        le = None

    if le is not None:
        idx2label = {i: c for i, c in enumerate(le.classes_)}
        print(f"[{path}] pakai label_encoder: {list(le.classes_)}")
    else:
        clf = pipe.named_steps.get("svm") or pipe[-1]
        classes = list(getattr(clf, "classes_", []))
        if not classes:
            raise RuntimeError(f"{path}: tidak ada encoder maupun classes_ di classifier")
        if all(isinstance(c, str) for c in classes):
            idx2label = {i: c for i, c in enumerate(classes)}
            print(f"[{path}] predict sudah string: {classes}")
        else:
            # asumsi: classes_ integer 0..n-1 → map ke L dengan urutan L
            # ⚠️ VERIFIKASI: cek manual beberapa sampel!
            idx2label = {int(c): L[int(c)] for c in classes}
            print(f"[{path}] classes_={classes} → mapping {idx2label} (VERIFIKASI!)")
    return pipe, idx2label


def make_pred(pipe, idx2label):
    def pred(X):
        p = np.asarray(pipe.predict(X))
        if np.issubdtype(p.dtype, np.integer):
            return np.array([idx2label[int(v)] for v in p])
        return p  # sudah string
    return pred


# ------------------------------------------------ lapor
def lapor(nama, csv, teks, label, pipe, idx2label):
    d = pd.read_csv(csv, encoding="utf-8-sig").dropna(subset=[teks, label])
    y = d[label].astype(str).str.strip().str.capitalize().to_numpy()

    # buang label di luar L
    mask = np.isin(y, L)
    if (~mask).any():
        print(f"⚠️ {nama}: buang {int((~mask).sum())} label di luar L: {set(y[~mask])}")
        d, y = d[mask], y[mask]

    pred = make_pred(pipe, idx2label)
    p = pred(d[teks].astype(str).to_numpy())

    # pastikan tipe konsisten
    assert p.dtype.kind in ("U", "O"), f"p masih {p.dtype}: {p[:3]}"

    print(f"\n== {nama} (n={len(y)}) ==")
    print("confusion matrix (baris=y, kolom=p):")
    print(confusion_matrix(y, p, labels=L))
    print(classification_report(y, p, labels=L, digits=3, zero_division=0))

    rng = np.random.default_rng(42)
    b = [f1_score(y[i], p[i], average="macro", labels=L, zero_division=0)
         for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    print("macro-F1 CI 95%:", np.percentile(b, [2.5, 97.5]).round(3))


# ------------------------------------------------ main
if __name__ == "__main__":
    pipe, idx2label = load_model("svm_pipeline.pkl")
    print(pipe)

    lapor("Test set",   "data/training/test_set_clean.csv", "clean_text", "label",
          pipe, idx2label)
    lapor("Sampel 200", "sampel_label_terisi.csv",             "Isi Ulasan", "label_manual",
          pipe, idx2label)