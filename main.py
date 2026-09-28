import json, re, joblib, nltk
import numpy as np
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Pipeline utuh (TF-IDF + SVM) — satu file, bukan dua terpisah lagi
pipeline = joblib.load('svm_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')
with open('slang_dict.json', encoding='utf-8') as f:
    slang_dict = json.load(f)

stop_words = set(nltk_stopwords.words('indonesian'))
# Disinkronkan dengan kata_penting versi training terbaru
kata_penting = {'tidak', 'belum', 'sangat', 'kurang', 'terlalu', 'sudah',
                'paling', 'lama', 'sekali', 'layanan', 'dan', 'di', 'oke'}
stop_words -= kata_penting
stemmer = StemmerFactory().create_stemmer()

ACCURACY = 0.9833
PRECISION_MACRO = 0.9477
RECALL_MACRO = 0.9715
F1_MACRO = 0.9590
CV_F1_MACRO = 0.8893

def full_preprocess(text):
    text = "" if not text else str(text)
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(slang_dict.get(w, w) for w in text.split())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

app = FastAPI()

class ReviewInput(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentimen: str
    confidence: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    cv_f1_macro: float

@app.post("/predict", response_model=PredictResponse)
def predict(data: ReviewInput):
    clean = full_preprocess(data.text)
    pred_int = pipeline.predict([clean])[0]
    label = label_encoder.inverse_transform([pred_int])[0]

    svm_step = pipeline.named_steps['svm']
    tfidf_step = pipeline.named_steps['tfidf']
    X_vec = tfidf_step.transform([clean])

    decision_scores = np.atleast_1d(svm_step.decision_function(X_vec)[0])
    if decision_scores.size > 1:
        sorted_scores = np.sort(decision_scores)
        confidence = float(sorted_scores[-1] - sorted_scores[-2])
    else:
        confidence = float(abs(decision_scores[0]))

    return {
        "sentimen": label,
        "confidence": round(confidence, 4),
        "accuracy": ACCURACY,
        "precision_macro": PRECISION_MACRO,
        "recall_macro": RECALL_MACRO,
        "f1_macro": F1_MACRO,
        "cv_f1_macro": CV_F1_MACRO
    }

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Sentimen API",
        version="1.0.0",
        routes=app.routes,
    )
    openapi_schema["openapi"] = "3.0.3"
    responses = openapi_schema["paths"]["/predict"]["post"]["responses"]
    responses.pop("422", None)
    schemas = openapi_schema["components"]["schemas"]
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
