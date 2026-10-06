"""
Preprocessing teks — SATU-SATUNYA sumber kebenaran, dipakai oleh:
- main.py                    (saat inference/predict)
- scripts/train_evaluate.py  (saat retrain)

Kalau logic preprocessing berubah, ubah di sini saja, supaya training dan
inference selalu konsisten.

Catatan: stopword removal SENGAJA dihapus. Daftar stopword NLTK Indonesia
membuang kata sentimen/negasi (baik, bukan, tak, jangan, tanpa, tapi, namun,
cukup, ...) dan terbukti menurunkan F1 macro.
"""
import json
import re

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def _ensure_nltk_data():
    for pkg, sub in [("punkt", "tokenizers"), ("punkt_tab", "tokenizers")]:
        try:
            nltk.data.find(f"{sub}/{pkg}")
        except LookupError:
            nltk.download(pkg)


_ensure_nltk_data()

_stemmer = StemmerFactory().create_stemmer()
_stem_cache = {}


def _stem(word):
    # Cache supaya kata yang sama tidak di-stem berulang (stemming Sastrawi lambat)
    if word not in _stem_cache:
        _stem_cache[word] = _stemmer.stem(word)
    return _stem_cache[word]


def load_slang_dict(path="slang_dict.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def full_preprocess(text, slang_dict):
    text = "" if not text else str(text)
    text = text.lower()
    text = re.sub(r"([a-z])\1{2,}", r"\1", text)   # "bagusss" -> "bagus"; angka tidak ikut (1000 tetap 1000)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(slang_dict.get(w, w) for w in text.split())
    tokens = word_tokenize(text)
    tokens = [_stem(w) for w in tokens]
    return " ".join(tokens)