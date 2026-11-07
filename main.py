"""main.py — API FastAPI Geosentimen RS Banyumas.
"""
import asyncio
import hmac
import json
import math
import os
import re
import time
import traceback
from collections import Counter, defaultdict
from typing import Optional
from fastapi.staticfiles import StaticFiles
import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from pydantic import BaseModel
from ops_agent import reviews
from sqlalchemy import insert
import uuid
from model_baru import bersihkan, muat_model_baru
from ops_agent import (OpsTools, make_engine, init_db, load_active_model,
                       export_reviews, run_routine, ops_running, agent_runs,
                       last_run_info, _susun_email,
                       reviews as REVIEWS_TBL, rs_info)

# Konfigurasi Groq untuk LLM ringkasan email
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
BASE_PAYLOAD = {
    "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "temperature": 0.2,
    "max_tokens": 1500
}

# 1. Inisialisasi Model & Metrics (menggunakan adapter model baru)
pipeline, label_encoder, _metrics = muat_model_baru("model_terbaik1.pkl", "metrics.json")

ENGINE = make_engine()
init_db(ENGINE)

_db_model = load_active_model(ENGINE)
if _db_model:
    pipeline, label_encoder, _metrics = _db_model
    print("[ops] memakai model dari database", flush=True)

ACCURACY = _metrics.get('accuracy', 0)
PRECISION_MACRO = _metrics.get('precision_macro', 0)
RECALL_MACRO = _metrics.get('recall_macro', 0)
F1_MACRO = _metrics.get('f1_macro', 0)
CV_F1_MACRO = _metrics.get('cv_f1_macro', 0)

# 2. Fungsi preprocess diarahkan ke bersihkan()
def preprocess(text):
    return bersihkan(text)

app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="static")

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def tangani_error(request: Request, exc: Exception):
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    resp = JSONResponse(status_code=500,
                        content={"detail": "Server mengalami kesalahan. Coba lagi sebentar lagi."})
    origin = request.headers.get("origin", "")
    if "*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin or "*"
    return resp


@app.get("/")
def health():
    return {"status": "ok"}


class ReviewInput(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentimen: str
    clean_text: str
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
        "cv_f1_macro": CV_F1_MACRO,
    }


# ============================================================
# Data RS untuk dashboard (tanpa chatbot)
# ============================================================
DATA_RS_PATH = os.environ.get("DATA_RS_PATH", "Data_RS_Banyumas.xlsx")
NEAR_KM = 15
OUT_OF_AREA_KM = 25

_STOP = {"yang", "dan", "untuk", "dengan", "dari", "atau", "adalah", "saya", "kamu",
         "anda", "rumah", "sakit", "ada", "apa", "bisa", "mau", "cari", "carikan",
         "tolong", "dong", "rekomendasi", "rekomendasikan", "banyumas", "purwokerto",
         "kabupaten", "rsud", "rsu", "hospital", "dekat", "terdekat", "daerah",
         "dimana", "mana", "berada", "lokasi", "tinggal"}
_KEC_AMBIGU = {"banyumas"}
_SINGKATAN = [(r"\bsel\b\.?", "selatan"), (r"\but\b\.?", "utara"),
              (r"\btim\b\.?", "timur"), (r"\bbar\b\.?", "barat")]


def _ncol(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _find_col(df, *names):
    cols = {_ncol(c): c for c in df.columns}
    for n in names:
        if _ncol(n) in cols:
            return cols[_ncol(n)]
    return None


def _to_float(v):
    try:
        f = float(str(v).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _kecamatan(addr):
    m = re.search(r"Kec\.?\s*([^,]+)", addr or "", flags=re.I)
    if not m:
        return ""
    k = m.group(1).strip()
    for pat, full in _SINGKATAN:
        k = re.sub(pat, full, k, flags=re.I)
    return " ".join(w.capitalize() for w in k.replace(".", " ").split())


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _tgl(v):
    try:
        import pandas as pd
        t = pd.to_datetime(v, errors="coerce", utc=True)
        return None if pd.isna(t) else t.strftime("%Y-%m-%d")
    except Exception:
        return None


def _load_rs_data(path):
    if not os.path.exists(path):
        print(f"[rs] {path} tidak ditemukan; lanjut tanpa data RS", flush=True)
        return []
    try:
        import pandas as pd
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        print(f"[rs] {path} tidak terbaca ({e}); lanjut tanpa data RS", flush=True)
        return []
    frames = [d for d in sheets.values() if _find_col(d, "Nama RS", "Nama Rumah Sakit")]
    if not frames:
        print("[rs] kolom 'Nama RS' tidak ditemukan", flush=True)
        return []
    df = pd.concat(frames, ignore_index=True)
    c_nama = _find_col(df, "Nama RS", "Nama Rumah Sakit")
    c_rat = _find_col(df, "Rating")
    c_addr = _find_col(df, "Lokasi Tempat", "Alamat", "Lokasi")
    c_lat = _find_col(df, "Latitude", "Lat")
    c_lon = _find_col(df, "Longitude", "Lng", "Lon", "Long")
    
    agg = defaultdict(lambda: {"n": 0, "rat": [], "addr": Counter(), "lat": [], "lon": []})
    for _, r in df.iterrows():
        nama = str(r[c_nama]).strip() if pd.notna(r[c_nama]) else ""
        if not nama or nama.lower() == "nan":
            continue
        h = agg[nama]
        h["n"] += 1
        rt = _to_float(r[c_rat]) if c_rat else None
        if rt is not None:
            h["rat"].append(rt)
        if c_addr and pd.notna(r[c_addr]) and str(r[c_addr]).strip():
            h["addr"][re.sub(r"\s+", " ", str(r[c_addr])).strip()] += 1
        la = _to_float(r[c_lat]) if c_lat else None
        lo = _to_float(r[c_lon]) if c_lon else None
        if la is not None and lo is not None and -90 <= la <= 90 and -180 <= lo <= 180:
            h["lat"].append(la)
            h["lon"].append(lo)

    hospitals = []
    for nama, h in agg.items():
        addr = h["addr"].most_common(1)[0][0] if h["addr"] else ""
        lat = sorted(h["lat"])[len(h["lat"]) // 2] if h["lat"] else None
        lon = sorted(h["lon"])[len(h["lon"]) // 2] if h["lon"] else None
        hospitals.append({"nama": nama, "alamat": addr, "kec": _kecamatan(addr),
                          "lat": lat, "lon": lon, "n": h["n"],
                          "rating": (sum(h["rat"]) / len(h["rat"])) if h["rat"] else None})
    print(f"[rs] data RS: {len(hospitals)} RS dari {df.shape[0]} ulasan", flush=True)
    return hospitals


RS_HOSPITALS = _load_rs_data(DATA_RS_PATH)
_BASE_HOSPITALS = RS_HOSPITALS
INFO = {}


def reload_rs_info():
    global INFO
    try:
        with ENGINE.connect() as c:
            rows = c.execute(select(rs_info)).mappings().all()
        INFO = {r["rs_name"]: {"phone": r["phone"], "website": r["website"],
                               "categories": json.loads(r["categories"] or "[]"),
                               "hours": json.loads(r["hours"] or "[]"),
                               "updated": r["updated_at"]} for r in rows}
    except Exception as e:
        print(f"[rs] info RS tidak dimuat: {e}", flush=True)
        INFO = {}


async def _run_dan_muat_ulang():
    # Panggil run_routine dengan argumen lengkap
    await run_routine(OPS, GROQ_URL, GROQ_KEY, BASE_PAYLOAD)
    try:
        await asyncio.to_thread(reload_rs_info)
    except Exception as e:
        print(f"[rs] muat ulang info gagal: {e}", flush=True)

try:
    reload_rs_info()
except Exception as e:
    print(f"[rs] gagal memuat info dari database: {e}", flush=True)


@app.get("/rs/profil")
def rs_profil(nama: str):
    cocok = [h for h in RS_HOSPITALS if nama.lower() in h["nama"].lower()]
    if not cocok:
        raise HTTPException(status_code=404, detail="RS tidak ditemukan di data.")
    hasil = []
    for h in cocok[:3]:
        i = INFO.get(h["nama"], {})
        hasil.append({"nama": h["nama"], "alamat": h["alamat"],
                      "kecamatan": h["kec"],
                      "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                      "jumlah_ulasan": h["n"], "telepon": i.get("phone"),
                      "website": i.get("website"), "jam_buka": i.get("hours") or []})
    return {"hasil": hasil}


@app.get("/rs/terdekat")
def rs_terdekat(daerah: str = "", lat: Optional[float] = None,
                lng: Optional[float] = None, batas: int = 5):
    origin = None
    if lat is not None and lng is not None and -90 <= lat <= 90 and -180 <= lng <= 180:
        origin = (lat, lng, "lokasi pengguna")
    if not origin:
        raise HTTPException(status_code=400, detail="Kirim lat dan lng.")
    ranked = sorted(
        ((_haversine(origin[0], origin[1], h["lat"], h["lon"]), h)
         for h in RS_HOSPITALS if h["lat"] is not None),
        key=lambda x: x[0])[:max(1, min(batas, 10))]
    peringatan = None
    if ranked and ranked[0][0] > OUT_OF_AREA_KM:
        peringatan = "Titik asal kemungkinan di luar Kabupaten Banyumas."
    return {
        "titik_acuan": origin[2],
        "catatan": "jarak_km = garis lurus (perkiraan)",
        "peringatan": peringatan,
        "hasil": [{"nama": h["nama"], "alamat": h["alamat"], "kecamatan": h["kec"],
                   "jarak_km": round(d, 1), "lebih_dari_15_km": d > NEAR_KM,
                   "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                   "jumlah_ulasan": h["n"]} for d, h in ranked],
    }


# ============================================================
# Ops agent
# ============================================================
def _deploy_model(pipe, le, metrics):
    global pipeline, label_encoder, ACCURACY, PRECISION_MACRO, RECALL_MACRO, \
        F1_MACRO, CV_F1_MACRO
    pipeline, label_encoder = pipe, le
    ACCURACY, PRECISION_MACRO = metrics["accuracy"], metrics["precision_macro"]
    RECALL_MACRO, F1_MACRO, CV_F1_MACRO = metrics["recall_macro"], metrics["f1_macro"], \
        metrics["cv_f1_macro"]


def _current_model():
    return pipeline, label_encoder, {"accuracy": ACCURACY,
                                     "precision_macro": PRECISION_MACRO,
                                     "recall_macro": RECALL_MACRO,
                                     "f1_macro": F1_MACRO,
                                     "cv_f1_macro": CV_F1_MACRO}


OPS = OpsTools(ENGINE, preprocess, _current_model, _deploy_model)

AGENT_RUN_TOKEN = os.environ.get("AGENT_RUN_TOKEN")


def _need_token(tok):
    if not AGENT_RUN_TOKEN or not hmac.compare_digest(
            (tok or "").encode(), AGENT_RUN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="token salah")


_retrain = {"berjalan": False, "hasil": None}


def _retrain_job():
    try:
        _retrain["hasil"] = OPS.call("retrain_model", {})
    finally:
        _retrain["berjalan"] = False


@app.post("/agent/run", status_code=202)
async def agent_run(background: BackgroundTasks,
                    x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    if ops_running():
        raise HTTPException(status_code=409, detail="rutinitas sedang berjalan")
    if _retrain["berjalan"]:
        raise HTTPException(status_code=409, detail="retrain sedang berjalan")
    background.add_task(_run_dan_muat_ulang)
    return {"status": "dimulai"}


@app.get("/agent/status")
def agent_status(x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    with ENGINE.connect() as c:
        runs = c.execute(select(agent_runs)
                         .order_by(agent_runs.c.started_at.desc()).limit(5)
                         ).mappings().all()
    return {"berjalan": ops_running(), "run_terakhir": [dict(r) for r in runs],
            "data": OPS.status_data()}


@app.get("/agent/ringkas")
def agent_ringkas(x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    return {"siklus_berjalan": ops_running(),
            "retrain_berjalan": _retrain["berjalan"],
            "run_terakhir": last_run_info(ENGINE), "data": OPS.status_data(),
            "hasil_retrain_terakhir": _retrain["hasil"]}


@app.post("/agent/retrain", status_code=202)
def agent_retrain(background: BackgroundTasks,
                  x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    if ops_running() or _retrain["berjalan"]:
        raise HTTPException(status_code=409, detail="rutinitas atau retrain sedang berjalan")
    _retrain["berjalan"] = True
    _retrain["hasil"] = None
    background.add_task(_retrain_job)
    return {"status": "dimulai"}


def _laporan_bulanan_job(periode=None):
    try:
        OPS.call("laporan_bulanan", {"periode": periode} if periode else {})
        OPS.call("kirim_laporan_bulanan", {"periode": periode} if periode else {})
    except Exception as e:
        print(f"[laporan] gagal: {e}", flush=True)


class LaporanInput(BaseModel):
    periode: Optional[str] = None


@app.post("/agent/laporan-bulanan", status_code=202)
def agent_laporan_bulanan(background: BackgroundTasks,
                          data: Optional[LaporanInput] = None,
                          x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    periode = (data.periode if data else None) or None
    background.add_task(_laporan_bulanan_job, periode)
    return {"status": "dimulai", "periode": periode or "bulan lalu"}


@app.get("/agent/laporan-bulanan")
def agent_laporan_bulanan_status(periode: str, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    from sqlalchemy import select as _sel
    from ops_agent import monthly_reports
    with ENGINE.connect() as c:
        rows = c.execute(_sel(monthly_reports)
                         .where(monthly_reports.c.period == periode)
                         .order_by(monthly_reports.c.negatif.desc())
                         ).mappings().all()
    return {"periode": periode, "jumlah": len(rows),
            "laporan": [{**dict(r),
                         "top_complaints": json.loads(r["top_complaints"] or "[]"),
                         "top_praises": json.loads(r["top_praises"] or "[]"),
                         "sample_negative": json.loads(r["sample_negative"] or "[]")}
                        for r in rows]}


@app.get("/health/agent")
def health_agent():
    s = OPS.status_kesehatan(float(os.environ.get("HEALTH_MAX_JAM", "13")))
    return JSONResponse(s, status_code=200 if s["ok"] else 503)


@app.get("/reviews.json")
def reviews_json():
    return export_reviews(ENGINE)


@app.get("/agent/ragu")
def agent_ragu(batas: int = 20, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    return OPS.lihat_ulasan_ragu(batas)


class KoreksiInput(BaseModel):
    review_id: str
    label: str


@app.post("/koreksi")
def koreksi(data: KoreksiInput, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    r = OPS.simpan_koreksi_manusia(data.review_id, data.label)
    if "error" in r:
        raise HTTPException(status_code=400, detail=r["error"])
    return r


@app.get("/ulasan/negatif-baru")
def ulasan_negatif_baru(batas: int = 20, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    return OPS.lihat_ulasan_negatif_belum_dikirim(batas)


@app.get("/ulasan/email-html")
def ulasan_email_html(x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    rows = OPS._pending_negative(50)
    if not rows:
        return {"subjek": "Tidak ada ulasan negatif baru",
                "html": "<p>Tidak ada ulasan negatif baru saat ini.</p>"}
    ringkasan = OPS._ringkasan_otomatis(rows)
    subjek, teks, html = _susun_email(rows, ringkasan)
    return {"subjek": subjek, "html": html}

class DummyInput(BaseModel):
    rs_name: str
    text: str
    rating: float

@app.post("/agent/inject-dummy")
def inject_dummy(data: DummyInput, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    rid = "dummy-" + uuid.uuid4().hex[:8]
    with ENGINE.begin() as c:
        c.execute(insert(reviews).values(
            review_id=rid,
            rs_name=data.rs_name,
            text=data.text,
            rating=data.rating,
            published_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            scraped_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            notified=0,
            legacy=0
        ))
    return {"status": "sukses", "review_id": rid}

# ============================================================
# Refresh manual
# ============================================================
REFRESH_COOLDOWN_SEC = int(os.environ.get("REFRESH_COOLDOWN_SEC", "3600"))
_last_refresh = 0.0


@app.post("/refresh")
async def refresh(background: BackgroundTasks):
    global _last_refresh
    if ops_running():
        raise HTTPException(status_code=409,
                            detail="Update sedang berjalan. Tunggu beberapa menit.")
    sisa = REFRESH_COOLDOWN_SEC - (time.time() - _last_refresh)
    if sisa > 0:
        raise HTTPException(status_code=429,
                            detail=f"Update baru saja dijalankan. Coba lagi sekitar "
                                   f"{int(sisa // 60) + 1} menit lagi.")
    _last_refresh = time.time()
    background.add_task(_run_dan_muat_ulang)
    return {"status": "dimulai"}


@app.get("/refresh/status")
def refresh_status():
    return last_run_info(ENGINE)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(title="Sentimen API", version="1.0.0",
                                 routes=app.routes)
    openapi_schema["openapi"] = "3.0.3"
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            operation.get("responses", {}).pop("422", None)
    schemas = openapi_schema.get("components", {}).get("schemas", {})
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi