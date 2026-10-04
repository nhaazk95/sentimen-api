"""
Preprocessing teks — SATU-SATUNYA sumber kebenaran, dipakai oleh:
- main.py            (saat inference/predict)
- scripts/train_evaluate.py  (saat retrain)

Kalau logic preprocessing berubah, ubah di sini saja, supaya training dan
inference selalu konsisten.
"""
import json
import re

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def _ensure_nltk_data():
    for pkg, sub in [("punkt", "tokenizers"), ("punkt_tab", "tokenizers"), ("stopwords", "corpora")]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            nltk.download(pkg)


_ensure_nltk_data()

# Disinkronkan dengan kata_penting versi training terbaru
KATA_PENTING = {
    "tidak", "belum", "sangat", "kurang", "terlalu", "sudah", "paling",
    "lama", "sekali", "layanan", "dan", "di", "oke", "ber", "ter", "ada",
}

_stop_words = set(nltk_stopwords.words("indonesian")) - KATA_PENTING
_stemmer = StemmerFactory().create_stemmer()


def load_slang_dict(path="slang_dict.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def full_preprocess(text, slang_dict):
    text = "" if not text else str(text)
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(slang_dict.get(w, w) for w in text.split())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in _stop_words]
    tokens = [_stemmer.stem(w) for w in tokens]
    return " ".join(tokens)
