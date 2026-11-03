import joblib, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import full_preprocess, load_slang_dict

pipe = joblib.load("svm_pipeline.pkl"); le = joblib.load("label_encoder.pkl")
slang = load_slang_dict("slang_dict.json")
tr = pd.read_csv("data/training/train_pool.csv"); te = pd.read_csv("data/training/test_set.csv")
print("kolom train_pool:", list(tr.columns))
print("label train_pool:", tr["label"].value_counts().to_dict())

norm = lambda s: s.astype(str).str.strip().str.lower()
print("teks persis sama:", int(norm(te["text"]).isin(set(norm(tr["text"]))).sum()))

ctr = tr["text"].apply(lambda t: full_preprocess(t, slang)); cte = te["text"].apply(lambda t: full_preprocess(t, slang))
print("teks bersih sama:", int(cte.isin(set(ctr)).sum()))

tf = pipe.named_steps["tfidf"]
sim = cosine_similarity(tf.transform(cte), tf.transform(ctr)).max(axis=1)
for th in (0.8, 0.9, 0.95):
    print(f"uji dengan kemiripan >= {th} terhadap train_pool:", int((sim >= th).sum()))