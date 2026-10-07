import joblib
from preprocessing import full_preprocess, load_slang_dict

slang = load_slang_dict("slang_dict.json")
pipe = joblib.load("svm_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

for t in ["Baik", "Bagus", "Good", "Bgus", "Sudah baik", "sudah cukup.",
          "Pelayanan judes banget", "Nunggu obat lama banget"]:
    print(f"{t!r:30} -> {le.inverse_transform(pipe.predict([full_preprocess(t, slang)]))[0]}")