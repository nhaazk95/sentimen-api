"""
Retrain SVM (GridSearchCV) pakai preprocessing & struktur pipeline yang SAMA
PERSIS dengan main.py (lihat scripts/preprocessing.py), lalu evaluasi ke
test_set.csv (label asli) dan dibandingkan dengan model yang sedang live
(root/metrics.json).

Output kandidat: data/models/candidate/svm_pipeline.pkl, label_encoder.pkl,
metrics.json, eval_predictions.csv. File-file ini baru dipindah ke root repo
oleh workflow, KALAU should_deploy == true.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from preprocessing import full_preprocess, load_slang_dict

TRAINING_DIR = "data/training"
CANDIDATE_DIR = "data/models/candidate"
BASELINE_METRICS_PATH = "metrics.json"  # metrics model yang SEDANG live di root repo
MIN_ACCURACY = 0.80  # ambang batas minimum; di bawah ini -> JANGAN deploy
VALID_LABELS = {"Positif", "Netral", "Negatif"}

# Nama step HARUS "tfidf" dan "svm" - main.py mengakses lewat
# pipeline.named_steps['svm'] dan pipeline.named_steps['tfidf']
PARAM_GRID = {
    "svm__C": [0.05, 0.1, 0.3, 1, 3],
}


def load_baseline_metrics():
    if not os.path.exists(BASELINE_METRICS_PATH):
        return {"accuracy": 0.0, "f1_macro": 0.0}
    with open(BASELINE_METRICS_PATH) as f:
        return json.load(f)


def main():
    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    slang_dict = load_slang_dict("slang_dict.json")

    train_pool = pd.read_csv(os.path.join(TRAINING_DIR, "train_pool.csv"))
    test_set = pd.read_csv(os.path.join(TRAINING_DIR, "test_set.csv"))

    print(f"[train_evaluate] Label unik di train_pool.csv: {train_pool['label'].unique().tolist()}")
    print(f"[train_evaluate] Label unik di test_set.csv  : {test_set['label'].unique().tolist()}")
    print(f"[train_evaluate] Baris train_pool: {len(train_pool)}, NaN label: {train_pool['label'].isna().sum()}")
    print(f"[train_evaluate] Baris test_set  : {len(test_set)}, NaN label: {test_set['label'].isna().sum()}")

    # Buang label kosong / di luar 3 kategori valid, SEBELUM preprocessing
    before_train, before_test = len(train_pool), len(test_set)
    train_pool = train_pool[train_pool["label"].isin(VALID_LABELS)].reset_index(drop=True)
    test_set = test_set[test_set["label"].isin(VALID_LABELS)].reset_index(drop=True)
    if len(train_pool) != before_train or len(test_set) != before_test:
        print(f"[train_evaluate] PERINGATAN: dibuang {before_train - len(train_pool)} baris invalid "
              f"dari train_pool, {before_test - len(test_set)} dari test_set.")

    # Bersihkan train: buang teks yang labelnya konflik, duplikat, dan yang bocor ke test
    n0 = len(train_pool)
    n_label = train_pool.groupby("text")["label"].transform("nunique")
    train_pool = train_pool[n_label == 1]
    train_pool = train_pool.drop_duplicates(subset="text")
    train_pool = train_pool[~train_pool["text"].isin(set(test_set["text"]))].reset_index(drop=True)
    print(f"[train_evaluate] Dibersihkan (konflik/duplikat/bocor ke test): {n0 - len(train_pool)} baris, sisa {len(train_pool)}")

    print("[train_evaluate] Preprocessing (slang normalize + stemming)...")
    X_train = train_pool["text"].apply(lambda t: full_preprocess(t, slang_dict))
    X_test = test_set["text"].apply(lambda t: full_preprocess(t, slang_dict))

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_pool["label"])
    y_test = label_encoder.transform(test_set["label"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)),
        ("svm", LinearSVC(class_weight="balanced", max_iter=10000)),
    ])

    print("[train_evaluate] Menjalankan GridSearchCV ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, PARAM_GRID, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    print(f"[train_evaluate] Best params: {grid.best_params_}")

    y_pred = best_pipeline.predict(X_test)
    new_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "cv_f1_macro": round(grid.best_score_, 4),
        "best_params": grid.best_params_,
        "n_train": len(train_pool),
    }

    # Confusion matrix + laporan per kelas
    labels_order = label_encoder.classes_
    print("[train_evaluate] Confusion matrix (baris=aktual, kolom=prediksi):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=labels_order, columns=labels_order))
    print("[train_evaluate] Classification report per kelas:")
    print(classification_report(y_test, y_pred, target_names=labels_order, zero_division=0))

    # Perbandingan baris-per-baris: label asli vs prediksi
    y_test_label = label_encoder.inverse_transform(y_test)
    y_pred_label = label_encoder.inverse_transform(y_pred)
    comparison = pd.DataFrame({
        "text": test_set["text"],
        "label_asli": y_test_label,
        "prediksi_model": y_pred_label,
        "cocok": y_test_label == y_pred_label,
    })
    comparison_path = os.path.join(CANDIDATE_DIR, "eval_predictions.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"[train_evaluate] Perbandingan per baris disimpan -> {comparison_path}")

    baseline = load_baseline_metrics()
    print(f"[train_evaluate] Baseline (live sekarang) : {baseline}")
    print(f"[train_evaluate] Kandidat                 : {new_metrics}")

    # Deploy kalau akurasi >= ambang DAN f1_macro tidak lebih buruk dari model live
    should_deploy = (
        new_metrics["accuracy"] >= MIN_ACCURACY
        and new_metrics["f1_macro"] >= baseline.get("f1_macro", 0.0)
    )

    joblib.dump(best_pipeline, os.path.join(CANDIDATE_DIR, "svm_pipeline.pkl"))
    joblib.dump(label_encoder, os.path.join(CANDIDATE_DIR, "label_encoder.pkl"))
    with open(os.path.join(CANDIDATE_DIR, "metrics.json"), "w") as f:
        json.dump(new_metrics, f, indent=2)

    summary = (
        f"Baseline (live sekarang) -> acc={baseline.get('accuracy')}, f1_macro={baseline.get('f1_macro')}\n"
        f"Kandidat -> acc={new_metrics['accuracy']}, f1_macro={new_metrics['f1_macro']}, "
        f"cv_f1_macro={new_metrics['cv_f1_macro']}\n"
        f"Syarat deploy: accuracy >= {MIN_ACCURACY} dan f1_macro >= baseline\n"
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