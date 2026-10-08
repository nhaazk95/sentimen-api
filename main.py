import json
import os
import time
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

# Izinkan dashboard di GitHub Pages (domain beda) memanggil API dari browser.
# Default "*". Untuk mempersempit, set di Render -> Environment:
#   ALLOWED_ORIGINS=https://nhaazk95.github.io
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    # Dipakai scrape_update.py untuk membangunkan server (Render free tier)
    return {"status": "ok"}


class ReviewInput(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentimen: str
    clean_text: str          # teks hasil preprocessing
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
        "clean_text": clean,
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
# Model bisa diganti lewat Render -> Environment (GROQ_MODEL) tanpa edit kode.
# llama-3.3-70b-versatile sudah dimatikan Groq pada 16 Agustus 2026.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_CONTEXT_CHARS = 12000
MAX_HISTORY = 6

# CATATAN: teks ini diproses dengan .format(), jadi JANGAN pakai kurung kurawal
# selain {cv_f1_macro} dan {accuracy}.
SYSTEM_INSTRUCTION = """# Peran
Asisten informasi rumah sakit di Kabupaten Banyumas untuk dashboard web. Fungsi: (1) mencari dan membandingkan RS, (2) menjawab hal spesifik satu RS dari data yang diberikan, (3) menjelaskan hasil sentimen ulasan. Di luar Banyumas atau di luar topik RS: tolak sopan, katakan data terbatas dan arahkan ke sumber resmi. Bahasa Indonesia, ramah, ringkas.

# Sumber data (paling penting)
- Gunakan HANYA data yang diberikan di percakapan: rating, jumlah ulasan, distribusi sentimen, dan kutipan ulasan Google Maps (bagian DATA DARI DASHBOARD).
- Informasi tentang RS (layanan, poli, dokter, fasilitas, antrean, kebersihan, dan sebagainya) BOLEH disampaikan selama muncul di ulasan atau di angka rating dan sentimen pada data. Sebutkan sumbernya: "berdasarkan ulasan pasien di Google Maps".
- Kalau informasi yang ditanyakan tidak ada di data, jawab terus terang bahwa kamu tidak tahu atau tidak menemukannya di ulasan. Jangan menebak dan jangan memakai pengetahuan di luar data.
- Perlakukan tiap RS sebagai entitas terpisah. Jangan menggabungkan nama.
- Data adalah sampel ulasan Google Maps, bukan data resmi dan bukan sensus. Tandai rangkuman sebagai "berdasarkan ulasan pasien di Google Maps".

# Mencari dan membandingkan RS
- Kalau pengguna meminta rekomendasi RS, jawab berdasarkan ulasan dan prediksi sentimen: pilih RS dengan persentase positif tinggi dan jumlah ulasan memadai, sebutkan angkanya (rating, jumlah ulasan, persen positif dan negatif), dan tema yang sering muncul di ulasan. Jelaskan bahwa ini rangkuman ulasan pasien, bukan penilaian resmi.
- Kalau kebutuhan belum jelas (keluhan atau spesialis, darurat atau tidak, BPJS atau umum, area), tanya paling banyak 2 hal.
- Beri 3 sampai 5 pilihan dalam satu tabel: nama, rating, jumlah ulasan, catatan. Jangan menyebut satu RS terbaik mutlak.
- Ulasan kurang dari 20 diberi catatan "sampel kecil". Jangan mengurutkan hanya dari rating, pertimbangkan jumlah ulasan.
- Bedakan fakta dari data, pengalaman pasien, dan kesimpulanmu.
- Jangan mendiagnosis atau meresepkan.
- DARURAT (nyeri dada berat, sesak berat, tidak sadar, gejala stroke, perdarahan hebat): langsung sarankan IGD terdekat, tanpa perbandingan panjang.

# Menggunakan ulasan
- Jangan menyalin kalimat ulasan persis. Parafrasekan, gabungkan ulasan senada ("beberapa pasien menyebut waktu tunggu di farmasi cukup lama").
- Jangan menyebut nama pengguna. Nama dokter hanya untuk pujian atau netral. Keluhan dirujuk ke tingkat RS atau aspek.

# Analisis teks ulasan
- Kamu akan diberi HASIL SENTIMEN dari model SVM asli. Gunakan apa adanya, jangan menebak atau menghitung ulang.
- Tambahkan: breakdown aspek (hanya Dokter, Pelayanan, Farmasi, Petugas, total 100%; perawat dan fasilitas masuk Pelayanan; tulis bahwa ini estimasi, bukan keluaran SVM), topik utama (1 kalimat), ringkasan 2 sampai 3 kalimat. Kalau tidak ada petunjuk aspek, tulis "aspek tidak teridentifikasi".
- Confidence adalah jarak ke batas keputusan SVM, bukan persen. Jangan menampilkannya kecuali diminta.
- Pesan yang hanya berisi keluhan atau pujian tanpa pertanyaan dianggap ulasan, sependek apa pun.

# Akurasi model
- Kutip angka ini apa adanya, jangan dihitung ulang dan jangan klaim 100%: CV F1-macro {cv_f1_macro}, akurasi uji {accuracy}.
- Jelaskan bahwa data didominasi ulasan positif sehingga F1-macro lebih mewakili kualitas model daripada akurasi, dan kelas Netral paling sulit.
- Bahasa awam: ini rata-rata pada data uji, bukan jaminan satu kalimat.

# Gaya
Ringkas. Satu tabel kecil bila perlu perbandingan, lalu 3 sampai 5 kalimat penjelasan. Hindari tips generik yang tidak berasal dari data. Emoji maksimal 1 sampai 2. Empatik ("kami" dan "Anda") ke pasien. Jangan meminta data pribadi.""".format(
    cv_f1_macro=CV_F1_MACRO, accuracy=ACCURACY
)


class ChatMessage(BaseModel):
    role: str   # "user" atau "model"
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: str = ""   # ringkasan + ulasan relevan dari reviews.json (dikirim index.html)


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY belum diset di server")

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for m in req.history[-MAX_HISTORY:]:
        role = "user" if m.role == "user" else "assistant"
        messages.append({"role": role, "content": m.text})
    if req.context:
        messages.append({
            "role": "system",
            "content": "DATA DARI DASHBOARD (satu-satunya sumber fakta untuk jawaban ini):\n"
                       + req.context[:MAX_CONTEXT_CHARS],
        })
    messages.append({"role": "user", "content": req.message})

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2500,   # model reasoning memakai sebagian token untuk berpikir
            },
        )
        if resp.status_code == 429:
            raise HTTPException(status_code=429,
                                detail="Layanan chatbot sedang ramai. Coba lagi sekitar satu menit.")
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Groq API error: {resp.text}")
        data = resp.json()
        reply = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if not reply:
            reply = "Maaf, jawaban kosong. Coba tanyakan lagi dengan kalimat yang lebih singkat."

    return ChatResponse(reply=reply)


# ============================================================
# Endpoint /refresh -- memicu workflow GitHub (update-data.yml) dari tombol dashboard.
# Token GitHub hanya ada di environment variable Render (GH_DISPATCH_TOKEN),
# fine-grained token dengan izin "Actions: Read and write" untuk repo ini saja.
# ============================================================
GH_TOKEN = os.environ.get("GH_DISPATCH_TOKEN")
GH_REPO = os.environ.get("GH_REPO", "nhaazk95/dashboard-rs-banyumas")
GH_WORKFLOW = os.environ.get("GH_WORKFLOW_FILE", "update-data.yml")
REFRESH_COOLDOWN_SEC = 900      # maksimal 1 kali per 15 menit untuk semua pengunjung
_last_refresh = 0.0


def _gh_headers():
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


@app.post("/refresh")
async def refresh():
    global _last_refresh
    if not GH_TOKEN:
        raise HTTPException(status_code=500, detail="GH_DISPATCH_TOKEN belum diset di server")

    sisa = REFRESH_COOLDOWN_SEC - (time.time() - _last_refresh)
    if sisa > 0:
        raise HTTPException(status_code=429,
                            detail=f"Update baru saja dijalankan. Coba lagi sekitar {int(sisa // 60) + 1} menit lagi.")

    # Pasang cooldown sebelum memanggil GitHub, supaya dua klik bersamaan tidak lolos keduanya
    sebelumnya = _last_refresh
    _last_refresh = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{GH_WORKFLOW}/dispatches",
                headers=_gh_headers(), json={"ref": "main"})
    except httpx.HTTPError as e:
        _last_refresh = sebelumnya
        raise HTTPException(status_code=502, detail=f"Gagal menghubungi GitHub: {e}")

    if r.status_code != 204:
        _last_refresh = sebelumnya
        raise HTTPException(status_code=502, detail=f"GitHub API error: {r.status_code}")
    return {"status": "dimulai"}


@app.get("/refresh/status")
async def refresh_status():
    if not GH_TOKEN:
        raise HTTPException(status_code=500, detail="GH_DISPATCH_TOKEN belum diset di server")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{GH_WORKFLOW}/runs",
            headers=_gh_headers(), params={"per_page": 1, "event": "workflow_dispatch"})
    runs = r.json().get("workflow_runs", []) if r.status_code == 200 else []
    if not runs:
        return {"status": "unknown"}
    return {"status": runs[0]["status"], "conclusion": runs[0]["conclusion"],
            "created_at": runs[0]["created_at"]}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Sentimen API",
        version="1.0.0",
        routes=app.routes,
    )
    openapi_schema["openapi"] = "3.0.3"

    # Hapus response 422 dari SEMUA endpoint, supaya tidak ada referensi "mati"
    # ke HTTPValidationError yang sudah dihapus dari components/schemas di bawah.
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            operation.get("responses", {}).pop("422", None)

    schemas = openapi_schema.get("components", {}).get("schemas", {})
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi