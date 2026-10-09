import json
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.insert(0, ".")
from preprocessing import full_preprocess, load_slang_dict

slang = load_slang_dict("slang_dict.json")
best = json.load(open("metrics.json"))["best_params"]
norm = lambda s: s.astype(str).str.lower().str.split().str.join(" ")

df = pd.read_csv("data/training/train_pool.csv")
df = df[df["label"].isin(["Positif", "Netral", "Negatif"])].copy()
df["_k"] = norm(df["text"])
df = df.sort_values("koreksi", ascending=False).drop_duplicates("_k")
test = pd.read_csv("data/training/test_set.csv")
df = df[~df["_k"].isin(set(norm(test["text"])))]


def undersample(d, ratio=5):
    cap = d["label"].value_counts().min() * ratio
    parts = []
    for _, g in d.groupby("label"):
        wajib = g[g["koreksi"] == 1]
        sisa = g[g["koreksi"] != 1]
        sisa = sisa.sample(n=min(len(sisa), max(cap - len(wajib), 0)), random_state=42)
        parts.append(pd.concat([wajib, sisa]))
    return pd.concat(parts, ignore_index=True)


def cv_report(d, nama):
    X = d["text"].apply(lambda t: full_preprocess(t, slang))
    y = d["label"]
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))), ("svm", SVC())])
    pipe.set_params(**best)
    pred = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"\n=== {nama}: n={len(d)} | F1 macro = {f1_score(y, pred, average='macro'):.4f}")
    print(classification_report(y, pred, digits=3))
    print(pd.crosstab(y, pred, rownames=["aktual"], colnames=["prediksi"]))


cv_report(undersample(df), "DENGAN koreksi")
cv_report(undersample(df[df["koreksi"] != 1]), "TANPA koreksi")