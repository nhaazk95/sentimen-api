"""
DIAGNOSTIK: evaluasi model yang SAAT INI live (svm_pipeline.pkl di root repo)
memakai data/training/test_set.csv yang sama dengan yang dipakai buat menilai
model kandidat hasil retrain.

Tujuannya: mastiin apakah baseline metrics di metrics.json (0.9833 / 0.959)
itu benar-benar sebanding kalau diuji di test_set.csv yang sekarang, atau
jangan-jangan angka itu dulu dihitung dari test set yang berbeda (lebih
besar/proporsinya beda), sehingga tidak apple-to-apple dibandingkan dengan
model kandidat baru.
"""
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from preprocessing import full_preprocess, load_slang_dict

pipeline = joblib.load("svm_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")
slang_dict = load_slang_dict("slang_dict.json")

test_set = pd.read_csv("data/training/test_set.csv")
X_test = test_set["text"].apply(lambda t: full_preprocess(t, slang_dict))
y_true = test_set["label"]

y_pred_int = pipeline.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_int)

print("=== Evaluasi MODEL YANG SEKARANG LIVE, di test_set.csv yang sama ===\n")
print(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision(m) : {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"Recall(m)    : {recall_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1 macro     : {f1_score(y_true, y_pred, average='macro'):.4f}")

print("\nConfusion matrix (baris=aktual, kolom=prediksi):")
labels_order = sorted(set(y_true) | set(y_pred))
cm = confusion_matrix(y_true, y_pred, labels=labels_order)
print(pd.DataFrame(cm, index=labels_order, columns=labels_order))

print("\nClassification report:")
print(classification_report(y_true, y_pred, labels=labels_order))
