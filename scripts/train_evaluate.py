"""
Retrain SVM (GridSearchCV) pakai preprocessing & struktur pipeline yang SAMA
PERSIS dengan main.py (lihat scripts/preprocessing.py), lalu evaluasi ke
test_set.csv (fixed) dan dibandingkan dengan model yang sedang live sekarang
(root/metrics.json).

Output kandidat: data/models/candidate/svm_pipeline.pkl, label_encoder.pkl,
metrics.json. File-file ini baru dipindah ke root repo (menggantikan yang
dipakai main.py) oleh workflow, KALAU should_deploy == true.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from preprocessing import full_preprocess, load_slang_dict

TRAINING_DIR = "data/training"
CANDIDATE_DIR = "data/models/candidate"
BASELINE_METRICS_PATH = "metrics.json"  # metrics model yang SEDANG live di root repo
TOLERANCE = 0.005  # kandidat boleh sedikit lebih rendah (0.5%) dan tetap lolos

# Nama step HARUS "tfidf" dan "svm" - main.py mengakses lewat
# pipeline.named_steps['svm'] dan pipeline.named_steps['tfidf']
PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1],
    "svm__kernel": ["rbf", "linear"],
}


def load_baseline_metrics():
    if not os.path.exists(BASELINE_METRICS_PATH):
        return {"accuracy": 0.0, "f1_macro": 0.0}
    with open(BASELINE_METRICS_PATH) as f:
        return json.load(f)


def main():
    slang_dict = load_slang_dict("slang_dict.json")

    train_pool = pd.read_csv(os.path.join(TRAINING_DIR, "train_pool.csv"))
    test_set = pd.read_csv(os.path.join(TRAINING_DIR, "test_set.csv"))

    print("[train_evaluate] Preprocessing (slang normalize + stemming)...")
    X_train = train_pool["text"].apply(lambda t: full_preprocess(t, slang_dict))
    X_test = test_set["text"].apply(lambda t: full_preprocess(t, slang_dict))

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_pool["label"])
    y_test = label_encoder.transform(test_set["label"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("svm", SVC()),
    ])

    print("[train_evaluate] Menjalankan GridSearchCV ...")
    grid = GridSearchCV(pipeline, PARAM_GRID, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    print(f"[train_evaluate] Best params: {grid.best_params_}")

    y_pred = best_pipeline.predict(X_test)
    new_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro"), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "cv_f1_macro": round(grid.best_score_, 4),
        "best_params": grid.best_params_,
        "n_train": len(train_pool),
    }

    baseline = load_baseline_metrics()
    print(f"[train_evaluate] Baseline (live sekarang): {baseline}")
    print(f"[train_evaluate] Kandidat                : {new_metrics}")

    should_deploy = (
        new_metrics["f1_macro"] >= baseline.get("f1_macro", 0) - TOLERANCE
        and new_metrics["accuracy"] >= baseline.get("accuracy", 0) - TOLERANCE
    )

    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(CANDIDATE_DIR, "svm_pipeline.pkl"))
    joblib.dump(label_encoder, os.path.join(CANDIDATE_DIR, "label_encoder.pkl"))
    with open(os.path.join(CANDIDATE_DIR, "metrics.json"), "w") as f:
        json.dump(new_metrics, f, indent=2)

    summary = (
        f"Baseline -> acc={baseline.get('accuracy')}, f1_macro={baseline.get('f1_macro')}\n"
        f"Kandidat -> acc={new_metrics['accuracy']}, f1_macro={new_metrics['f1_macro']}, "
        f"cv_f1_macro={new_metrics['cv_f1_macro']}\n"
        f"Keputusan: {'DEPLOY' if should_deploy else 'JANGAN DEPLOY, investigasi dulu'}"
    )
    print("[train_evaluate]\n" + summary)

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"should_deploy={'true' if should_deploy else 'false'}\n")
            f.write("summary<<EOF\n" + summary + "\nEOF\n")


if __name__ == "__main__":
    main()
