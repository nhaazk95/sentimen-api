# cek_encoder.py
import joblib
le = joblib.load("label_encoder_final.pkl")
print("Kelas:", le.classes_)
print("Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))