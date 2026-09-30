import json
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from preprocessing import full_preprocess, load_slang_dict  # <- diambil dari modul bersama

# Pipeline utuh (TF-IDF + SVM) — satu file
pipeline = joblib.load('svm_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')
slang_dict = load_slang_dict('slang_dict.json')

# Metrics sekarang dibaca dari file, bukan hardcoded, supaya otomatis
# ter-update tiap kali pipeline retraining deploy model baru.
with open('metrics.json', encoding='utf-8') as f:
    _metrics = json.load(f)

ACCURACY = _metrics['accuracy']
PRECISION_MACRO = _metrics['precision_macro']
RECALL_MACRO = _metrics['recall_macro']
F1_MACRO = _metrics['f1_macro']
CV_F1_MACRO = _metrics['cv_f1_macro']


def preprocess(text):
    return full_preprocess(text, slang_dict)


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
    clean = preprocess(data.text)
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
