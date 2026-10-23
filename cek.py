import joblib, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import full_preprocess, load_slang_dict

pipe = joblib.load("svm_pipeline.pkl")
le = joblib.load("label_encoder.pkl")
slang = load_slang_dict("slang_dict.json")

df = pd.read_csv("data/training/test_set.csv")   # ganti sesuai data uji yang dievaluasi
df = df[df["label"].isin(["Positif", "Netral", "Negatif"])].reset_index(drop=True)
X = df["text"].apply(lambda t: full_preprocess(t, slang))
y_true, y_pred = le.transform(df["label"]), pipe.predict(X)

print("jumlah baris:", len(df), "| accuracy:", round(accuracy_score(y_true, y_pred), 4))
print(classification_report(y_true, y_pred, target_names=le.classes_, digits=4))
print(confusion_matrix(y_true, y_pred))