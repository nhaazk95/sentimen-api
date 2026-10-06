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
MIN_ACCURACY = 0.98  # ambang batas tetap: di bawah ini -> JANGAN deploy, buka Issue

# Undersampling moderat kelas mayoritas (Positif), BUKAN disamain rata (itu
# bakal buang terlalu banyak data). Kelas mayoritas dibatasi maks N kali
# lipat dari kelas minoritas terkecil. Set UNDERSAMPLE=False buat matiin.
UNDERSAMPLE = True
MAX_RATIO_TO_MINORITY = 5

# Nama step HARUS "tfidf" dan "svm" - main.py mengakses lewat
# pipeline.named_steps['svm'] dan pipeline.named_steps['tfidf']
PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.01, 0.1, 1],
    "svm__kernel": ["rbf", "linear"],
    "svm__class_weight": [None, "balanced"],  # penting krn data timpang (~82% Positif)
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

    # --- DEBUG: pastiin cuma ada 3 label yang valid ---------------------
    print(f"[train_evaluate] Label unik di train_pool.csv: {train_pool['label'].unique().tolist()}")
    print(f"[train_evaluate] Label unik di test_set.csv  : {test_set['label'].unique().tolist()}")
    print(f"[train_evaluate] Jumlah baris train_pool: {len(train_pool)}, NaN label: {train_pool['label'].isna().sum()}")
    print(f"[train_evaluate] Jumlah baris test_set  : {len(test_set)}, NaN label: {test_set['label'].isna().sum()}")
    # ----------------------------------------------------------------------

    # Buang baris dengan label kosong/NaN atau di luar 3 kategori valid,
    # SEBELUM preprocessing teks, supaya X dan y tetap sinkron panjangnya.
    VALID_LABELS = {"Positif", "Netral", "Negatif"}
    before_train, before_test = len(train_pool), len(test_set)
    train_pool = train_pool[train_pool["label"].isin(VALID_LABELS)].reset_index(drop=True)
    test_set = test_set[test_set["label"].isin(VALID_LABELS)].reset_index(drop=True)
    if len(train_pool) != before_train or len(test_set) != before_test:
        print(f"[train_evaluate] PERINGATAN: dibuang {before_train - len(train_pool)} baris invalid "
              f"dari train_pool, {before_test - len(test_set)} dari test_set (label di luar 3 kategori valid).")

    if UNDERSAMPLE:
        if "koreksi" not in train_pool.columns:
            train_pool["koreksi"] = 0
        train_pool["koreksi"] = train_pool["koreksi"].fillna(0).astype(int)

        counts = train_pool["label"].value_counts()
        cap = counts.min() * MAX_RATIO_TO_MINORITY

        parts = []
        for lbl, group in train_pool.groupby("label"):
            wajib = group[group["koreksi"] == 1]       # selalu ikut
            sisa = group[group["koreksi"] != 1]
            slot = max(cap - len(wajib), 0)
            if len(sisa) > slot:
                sisa = sisa.sample(n=slot, random_state=42)
            parts.append(pd.concat([wajib, sisa]))
        train_pool = pd.concat(parts, ignore_index=True)

    # Pakai "text_processed" kalau udah ada (diproses dari Workflow 1), biar
    # gak diproses dua kali. Baris lama yang belum punya kolom ini (sebelum
    # fitur ini ditambahkan) tetap diproses di sini sebagai fallback.
    print("[train_evaluate] Menyiapkan teks (pakai text_processed kalau ada)...")
    if "text_processed" in train_pool.columns:
        needs_processing = train_pool["text_processed"].isna()
        train_pool.loc[needs_processing, "text_processed"] = train_pool.loc[needs_processing, "text"].apply(
            lambda t: full_preprocess(t, slang_dict)
        )
        X_train = train_pool["text_processed"]
    else:
        X_train = train_pool["text"].apply(lambda t: full_preprocess(t, slang_dict))

    if "text_processed" in test_set.columns:
        needs_processing = test_set["text_processed"].isna()
        test_set.loc[needs_processing, "text_processed"] = test_set.loc[needs_processing, "text"].apply(
            lambda t: full_preprocess(t, slang_dict)
        )
        X_test = test_set["text_processed"]
    else:
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

    # Confusion matrix + per-kelas breakdown, supaya kalau gagal lolos evaluasi,
    # kelihatan kelas mana yang jadi biang keroknya (biasanya Negatif/Netral).
    from sklearn.metrics import classification_report, confusion_matrix
    labels_order = label_encoder.classes_
    print("[train_evaluate] Confusion matrix (baris=aktual, kolom=prediksi):")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=labels_order, columns=labels_order))
    print("[train_evaluate] Classification report per kelas:")
    print(classification_report(y_test, y_pred, target_names=labels_order))

    # Simpan perbandingan baris-per-baris: teks asli, label (manual/rating),
    # vs prediksi model kandidat - buat lihat contoh konkret yang salah.
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
    print(f"[train_evaluate] Syarat deploy: accuracy >= {MIN_ACCURACY}")

    should_deploy = new_metrics["accuracy"] >= MIN_ACCURACY

    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(CANDIDATE_DIR, "svm_pipeline.pkl"))
    joblib.dump(label_encoder, os.path.join(CANDIDATE_DIR, "label_encoder.pkl"))
    with open(os.path.join(CANDIDATE_DIR, "metrics.json"), "w") as f:
        json.dump(new_metrics, f, indent=2)

    summary = (
        f"Baseline (live sekarang) -> acc={baseline.get('accuracy')}, f1_macro={baseline.get('f1_macro')}\n"
        f"Kandidat -> acc={new_metrics['accuracy']}, f1_macro={new_metrics['f1_macro']}, "
        f"cv_f1_macro={new_metrics['cv_f1_macro']}\n"
        f"Syarat deploy: accuracy >= {MIN_ACCURACY}\n"
        f"Keputusan: {'DEPLOY' if should_deploy else f'JANGAN DEPLOY (accuracy di bawah {MIN_ACCURACY}), investigasi dulu'}"
    )
    print("[train_evaluate]\n" + summary)

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"should_deploy={'true' if should_deploy else 'false'}\n")
            f.write("summary<<EOF\n" + summary + "\nEOF\n")


if __name__ == "__main__":
    main()