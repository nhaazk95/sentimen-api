import json
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.insert(0, ".")
from preprocessing import full_preprocess, load_slang_dict

slang = load_slang_dict("slang_dict.json")
best = json.load(open("metrics.json"))["best_params"]
print("best_params:", best)
norm = lambda s: s.astype(str).str.lower().str.split().str.join(" ")

df = pd.read_csv("data/training/train_pool.csv")
df = df[df["label"].isin(["Positif", "Netral", "Negatif"])].copy()
df["_k"] = norm(df["text"])
df = df.sort_values("koreksi", ascending=False).drop_duplicates("_k")
df = df[~df["_k"].isin(set(norm(pd.read_csv("data/training/test_set.csv")["text"])))]
df["clean"] = df["text"].apply(lambda t: full_preprocess(t, slang))   # sekali saja, stemming lambat


def undersample(d, ratio):
    cap = d["label"].value_counts().min() * ratio
    parts = []
    for _, g in d.groupby("label"):
        wajib, sisa = g[g["koreksi"] == 1], g[g["koreksi"] != 1]
        sisa = sisa.sample(n=min(len(sisa), max(cap - len(wajib), 0)), random_state=42)
        parts.append(pd.concat([wajib, sisa]))
    return pd.concat(parts, ignore_index=True)


baris = []
for ratio in (3, 5, 8):
    d = undersample(df, ratio)
    for cw in (None, "balanced"):
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))), ("svm", SVC())])
        pipe.set_params(**{**best, "svm__class_weight": cw})
        pred = cross_val_predict(pipe, d["clean"], d["label"], cv=StratifiedKFold(5, shuffle=True, random_state=42))
        p, r, f, _ = precision_recall_fscore_support(d["label"], pred, labels=["Netral"], zero_division=0)
        baris.append({"rasio": ratio, "class_weight": cw, "n": len(d),
                      "F1_macro": round(f1_score(d["label"], pred, average="macro"), 3),
                      "Netral_P": round(p[0], 3), "Netral_R": round(r[0], 3), "Netral_F1": round(f[0], 3)})
        if ratio == 5 and cw == best.get("svm__class_weight"):
            x = d.assign(prediksi=pred)
            x[x["label"] != x["prediksi"]][["text", "label", "prediksi", "koreksi"]].to_csv("salah_cv.csv", index=False)

print(pd.DataFrame(baris).to_string(index=False))
print("\nContoh yang salah disimpan di salah_cv.csv")