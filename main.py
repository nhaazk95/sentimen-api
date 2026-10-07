import json
import os
import joblib
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

from preprocessing import full_preprocess, load_slang_dict  # modul bersama

# Pipeline utuh (TF-IDF + SVM) -- satu file
pipeline = joblib.load('svm_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')
slang_dict = load_slang_dict('slang_dict.json')

# Metrics dibaca dari file supaya otomatis ter-update saat retraining
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

# Izinkan dashboard di GitHub Pages (domain beda) memanggil API dari browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # bisa dipersempit ke "https://nhaazk95.github.io"
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewInput(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentimen: str
    clean_text: str          # BARU: teks hasil preprocessing
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
        "clean_text": clean,   # BARU
        "confidence": round(confidence, 4),
        "accuracy": ACCURACY,
        "precision_macro": PRECISION_MACRO,
        "recall_macro": RECALL_MACRO,
        "f1_macro": F1_MACRO,
        "cv_f1_macro": CV_F1_MACRO
    }


# ============================================================
# Endpoint /chat -- proxy ke Groq, API key hanya ada di
# environment variable server Render.
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # WAJIB diset di Render -> Environment
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_INSTRUCTION = """Kamu adalah asisten analisis rumah sakit di Kabupaten Banyumas untuk sebuah dashboard web.

Tugasmu:
1. Membantu mencari/membandingkan RS berdasarkan data yang diberikan (rating, jumlah ulasan,
   distribusi sentimen). Jangan menyebut satu RS sebagai "terbaik" mutlak -- beri beberapa
   pilihan sesuai kondisi, urgensi, dan preferensi pengguna.
2. Kalau user menempelkan teks ulasan untuk dianalisis, kamu akan diberi HASIL SENTIMEN dari
   model SVM asli (bukan dari dirimu sendiri) -- gunakan itu apa adanya, JANGAN menebak sentimen
   sendiri. Tambahkan breakdown aspek (HANYA dari 4 kategori: Dokter, Pelayanan, Farmasi, Petugas
   -- aspek lain seperti perawat/fasilitas masuk ke kategori "Pelayanan"), topik utama (1 kalimat),
   dan ringkasan singkat (2-3 kalimat).
3. Kalau ditanya akurasi model, kutip angka tetap berikut (jangan menghitung/mengarang ulang):
   CV F1-macro {cv_f1_macro}, akurasi uji manual (test set terpisah tanpa overlap dari training): {accuracy}.
4. Kondisi darurat (nyeri dada berat, sesak berat, penurunan kesadaran, gejala stroke): prioritaskan
   keselamatan, sarankan segera ke IGD terdekat, jangan tunda dengan perbandingan panjang.
5. Jangan mendiagnosis atau menggantikan saran tenaga medis.

Gunakan Bahasa Indonesia yang jelas, ringkas, dan netral.""".format(
    cv_f1_macro=CV_F1_MACRO, accuracy=ACCURACY
)


class ChatMessage(BaseModel):
    role: str   # "user" atau "model"
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY belum diset di server")

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for m in req.history:
        role = "user" if m.role == "user" else "assistant"
        messages.append({"role": role, "content": m.text})
    messages.append({"role": "user", "content": req.message})

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "openai/gpt-oss-120b", "messages": messages},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq API error: {resp.text}")
        data = resp.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "Maaf, terjadi kesalahan.")

    return ChatResponse(reply=reply)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Sentimen API",
        version="1.0.0",
        routes=app.routes,
    )
    openapi_schema["openapi"] = "3.0.3"

    # FIX: hapus response 422 dari SEMUA endpoint, bukan cuma /predict,
    # supaya tidak ada lagi referensi "mati" ke HTTPValidationError
    # yang sudah dihapus dari components/schemas di bawah.
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            operation.get("responses", {}).pop("422", None)

    schemas = openapi_schema.get("components", {}).get("schemas", {})
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi