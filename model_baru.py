"""model_baru.py - memasang model_terbaik1.pkl ke main.py tanpa mengubah banyak kode.

Model baru: TF-IDF (kata + karakter) -> SVM linear, label keluaran berupa teks huruf kecil
('negatif' | 'netral' | 'positif'), langkah pipeline bernama 'tfidf' dan 'clf'.
main.py lama mengharapkan: label integer + label_encoder, dan langkah bernama 'svm'.
Adapter di bawah menjembatani keduanya, sehingga /predict dan ops_agent yang memakai
tiga serangkai (pipeline, label_encoder, metrics) tetap berjalan.

PENTING: bersihkan() HARUS identik dengan fungsi saat pelatihan (bandingkanmodel.py, USE_STEM=False).
"""
import json
import re
import unicodedata

import joblib
import numpy as np

# ---------- SALINAN PERSIS bersihkan() dari bandingkanmodel.py ----------
EMO_POS = r"[👍👌🌟⭐✨❤♥💗💖💓💙🩷🤩😍🥰🫶🔥💯]"
EMO_NEG = r"[👎😡🤬]"


def bersihkan(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\\[rn]|<br>", " ", s)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:kelamaan|lama)\b", " cepat ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:lambat|lelet)\b", " cepat ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:siang)\b", " pagi ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:membedabedakan)\b", " adil ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:ada\s+)?(?:kekurangan|kurangnya|kurang|masalah)\b", " sempurna ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:jauh|lama|lelet|mahal|ribet|kecewa|mengecewakan)\b", " bagus ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:usah|perlu)\s+bawa\b", " bagus ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+bakal\s+lupain?\b", " suka ", s)
    s = re.sub(r"\b(?:memeuaskann|memeuaskan|muas|muaskn|memuaskan)\b", " puas ", s)
    s = re.sub(r"\b(?:bgs|bgus|bagus|mantap|mantul|keren)\b", " bagus ", s)
    s = re.sub(EMO_POS, " emopos ", s)
    s = re.sub(EMO_NEG, " emoneg ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- adapter: meniru antarmuka LabelEncoder & pipeline lama ----------
class EncoderKompat:
    """Meniru LabelEncoder lama: 0/1/2 <-> 'Negatif'/'Netral'/'Positif' (huruf kapital seperti sebelumnya)."""
    classes_ = np.array(["Negatif", "Netral", "Positif"], dtype=object)

    def transform(self, y):
        daftar = list(self.classes_)
        return np.array([daftar.index(str(v).capitalize()) for v in y])

    def inverse_transform(self, y):
        return np.array([self.classes_[int(i)] for i in y], dtype=object)


class ModelBaru:
    """Membungkus pipeline model_terbaik1.pkl; predict() mengembalikan kode integer (0/1/2)."""

    def __init__(self, pipe):
        self._pipe = pipe
        urut = [str(c).capitalize() for c in pipe.classes_]
        assert urut == list(EncoderKompat.classes_), f"Urutan kelas tak sesuai: {urut}"
        self.classes_ = np.arange(len(urut))
        # nama langkah lama: 'tfidf' dan 'svm' (di model baru: 'tfidf' dan 'clf')
        self.named_steps = {"tfidf": pipe.named_steps["tfidf"], "svm": pipe.named_steps["clf"]}

    def predict(self, X):
        idx = {c: i for i, c in enumerate(self._pipe.classes_)}
        return np.array([idx[v] for v in self._pipe.predict(X)])

    def predict_proba(self, X):
        return self._pipe.predict_proba(X)

    def decision_function(self, X):
        return self._pipe.decision_function(X)


def muat_model_baru(path_model="model_terbaik1.pkl", path_metrics="metrics.json"):
    """Kembalikan (pipeline_kompat, label_encoder_kompat, metrics) seperti tiga serangkai lama."""
    pipe = joblib.load(path_model)
    with open(path_metrics, encoding="utf-8") as f:
        metrics = json.load(f)
    return ModelBaru(pipe), EncoderKompat(), metrics