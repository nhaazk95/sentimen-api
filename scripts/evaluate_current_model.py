import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

import os
import sys

# preprocessing.py ada di ROOT repo (satu-satunya sumber, dipakai juga oleh main.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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