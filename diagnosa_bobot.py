import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.insert(0, ".")
from preprocessing import full_preprocess, load_slang_dict

LAB = ["Negatif", "Netral", "Positif"]
slang = load_slang_dict("slang_dict.json")
best = json.load(open("metrics.json"))["best_params"]
norm = lambda s: s.astype(str).str.lower().str.split().str.join(" ")

df = pd.read_csv("data/training/train_pool.csv")
df = df[df["label"].isin(LAB)].copy()
df["_k"] = norm(df["text"])
df = df.sort_values("koreksi", ascending=False).drop_duplicates("_k")
df = df[~df["_k"].isin(set(norm(pd.read_csv("data/training/test_set.csv")["text"])))]
nyata = df["label"].value_counts().reindex(LAB)          # distribusi asli


def undersample(d, ratio=5):
    cap = d["label"].value_counts().min() * ratio
    parts = []
    for _, g in d.groupby("label"):
        wajib, sisa = g[g["koreksi"] == 1], g[g["koreksi"] != 1]
        sisa = sisa.sample(n=min(len(sisa), max(cap - len(wajib), 0)), random_state=42)
        parts.append(pd.concat([wajib, sisa]))
    return pd.concat(parts, ignore_index=True)


d = undersample(df)
d["clean"] = d["text"].apply(lambda t: full_preprocess(t, slang))
sampel = d["label"].value_counts().reindex(LAB)
skala = (nyata / sampel).values[:, None]                  # skala baris ke distribusi asli


def prf(cm):
    tp = np.diag(cm)
    P = tp / np.maximum(cm.sum(0), 1)
    R = tp / np.maximum(cm.sum(1), 1)
    return P, R, 2 * P * R / np.maximum(P + R, 1e-9)


konfig = [("balanced", "balanced")] + [
    (f"Netral x{w}", {"Negatif": 1.0, "Netral": w, "Positif": 1.0}) for w in (1, 1.5, 2, 3)]
hasil = []
for nama, cw in konfig:
    m = {k: [] for k in ("fm", "nP", "nR", "nF", "fm_real", "nP_real", "nF_real")}
    for seed in (1, 2, 3):
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))), ("svm", SVC())])
        pipe.set_params(**{**best, "svm__class_weight": cw})
        pred = cross_val_predict(pipe, d["clean"], d["label"], cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        cm = confusion_matrix(d["label"], pred, labels=LAB)
        P, R, F = prf(cm)
        P2, R2, F2 = prf(cm * skala)
        m["fm"].append(F.mean()); m["nP"].append(P[1]); m["nR"].append(R[1]); m["nF"].append(F[1])
        m["fm_real"].append(F2.mean()); m["nP_real"].append(P2[1]); m["nF_real"].append(F2[1])
    hasil.append({"konfigurasi": nama,
                  "F1_macro": f"{np.mean(m['fm']):.3f}±{np.std(m['fm']):.3f}",
                  "Netral_P": round(np.mean(m["nP"]), 3), "Netral_R": round(np.mean(m["nR"]), 3),
                  "Netral_F1": round(np.mean(m["nF"]), 3),
                  "F1_macro_asli": round(np.mean(m["fm_real"]), 3),
                  "Netral_P_asli": round(np.mean(m["nP_real"]), 3),
                  "Netral_F1_asli": round(np.mean(m["nF_real"]), 3)})
print(pd.DataFrame(hasil).to_string(index=False))
print("\nDistribusi asli:", nyata.to_dict(), "| sampel CV:", sampel.to_dict())