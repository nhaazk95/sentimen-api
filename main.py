import asyncio
import difflib
import hmac
import json
import math
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from typing import Optional

import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select

from preprocessing import full_preprocess, load_slang_dict
from ops_agent import (OpsTools, make_engine, init_db, load_active_model, export_reviews,
                       run_routine, ops_running, agent_runs, last_run_info, _susun_email,
                       reviews as REVIEWS_TBL, rs_info)
# ============================================================
# PERBAIKAN: Memuat model final (hasil eksperimen 0.7837)
# ============================================================
pipeline = joblib.load('svm_pipeline_final.pkl')
label_encoder = joblib.load('label_encoder_final.pkl')
slang_dict = load_slang_dict('slang_dict.json')

# Metrics dibaca dari file supaya otomatis ter-update saat retraining
with open('metrics.json', encoding='utf-8') as f:
    _metrics = json.load(f)

ENGINE = make_engine()
init_db(ENGINE)
_db_model = load_active_model(ENGINE)
if _db_model:                       # model hasil retrain menggantikan file .pkl dari repo
    pipeline, label_encoder, _metrics = _db_model
    print("[ops] memakai model dari database", flush=True)

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
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def tangani_error(request: Request, exc: Exception):
    traceback.print_exception(type(exc), exc, exc.__traceback__)       # penyebab asli masuk ke Logs Render
    resp = JSONResponse(status_code=500, content={"detail": "Server mengalami kesalahan. Coba lagi sebentar lagi."})
    origin = request.headers.get("origin", "")
    if "*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin or "*"
    return resp


@app.get("/")
def health():
    # Dipakai untuk membangunkan server (Render free tier) dan ping berkala
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
# Data_RS_Banyumas.xlsx -- satu baris = satu ULASAN. Dari file ini (digabung ulasan baru dari database)
# dibangun: profil per RS, jarak dari kecamatan/koordinat, dan pencarian kutipan ulasan.
# Dipakai oleh endpoint /rs/* (tool agent Copilot Studio).
# ============================================================
DATA_RS_PATH = os.environ.get("DATA_RS_PATH", "Data_RS_Banyumas.xlsx")
MAX_REVIEW_SNIPPETS = 12      # kutipan ulasan relevan yang dikembalikan
SNIPPET_CHARS = 220
NEAR_KM = 15                  # di atas jarak ini RS ditandai "jauh"
OUT_OF_AREA_KM = 25           # titik asal yang lebih jauh dari ini dari RS terdekat dianggap di luar Banyumas

_STOP = {"yang", "dan", "untuk", "dengan", "dari", "atau", "adalah", "saya", "kamu", "anda", "rumah", "sakit",
         "ada", "apa", "bisa", "mau", "cari", "carikan", "tolong", "dong", "rekomendasi", "rekomendasikan",
         "banyumas", "purwokerto", "kabupaten", "rsud", "rsu", "hospital", "dekat", "terdekat", "daerah",
         "dimana", "mana", "berada", "lokasi", "tinggal"}
_KEC_AMBIGU = {"banyumas"}    # kecamatan yang namanya sama dengan kabupaten: hanya dipakai bila ditulis "kecamatan banyumas"
_SINGKATAN = [(r"\bsel\b\.?", "selatan"), (r"\but\b\.?", "utara"), (r"\btim\b\.?", "timur"), (r"\bbar\b\.?", "barat")]


def _ncol(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _find_col(df, *names):
    cols = {_ncol(c): c for c in df.columns}
    for n in names:
        if _ncol(n) in cols:
            return cols[_ncol(n)]
    return None


def _to_float(v):
    """Menerima angka biasa maupun desimal koma ("-7,5180652")."""
    try:
        f = float(str(v).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _kecamatan(addr):
    """'... Kec. Purwokerto Sel., Kabupaten ...' -> 'Purwokerto Selatan' (singkatan arah dilengkapi)."""
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
    """Mengembalikan (hospitals, reviews). Keduanya list kosong bila file tidak bisa dipakai."""
    if not os.path.exists(path):
        print(f"[rs] {path} tidak ditemukan; lanjut tanpa data RS", flush=True)
        return [], []
    try:
        import pandas as pd
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:  # openpyxl belum terpasang, file rusak, dll.
        print(f"[rs] {path} tidak terbaca ({e}); lanjut tanpa data RS", flush=True)
        return [], []
    frames = [d for d in sheets.values() if _find_col(d, "Nama RS", "Nama Rumah Sakit")]
    if not frames:
        print("[rs] kolom 'Nama RS' tidak ditemukan; lanjut tanpa data RS", flush=True)
        return [], []
    df = pd.concat(frames, ignore_index=True)
    c_nama = _find_col(df, "Nama RS", "Nama Rumah Sakit")
    c_rat = _find_col(df, "Rating")
    c_addr = _find_col(df, "Lokasi Tempat", "Alamat", "Lokasi")
    c_lat = _find_col(df, "Latitude", "Lat")
    c_lon = _find_col(df, "Longitude", "Lng", "Lon", "Long")
    c_text = _find_col(df, "Isi Ulasan", "Ulasan", "Review")
    c_clean = _find_col(df, "clean_text")
    c_date = _find_col(df, "Waktu Ulasan", "Tanggal", "Published At")

    agg = defaultdict(lambda: {"n": 0, "rat": [], "addr": Counter(), "lat": [], "lon": []})
    reviews = []
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
        if c_text and c_clean and pd.notna(r[c_text]) and pd.notna(r[c_clean]):
            txt = re.sub(r"\s+", " ", str(r[c_text])).strip()
            tok = set(str(r[c_clean]).split())
            if len(txt) >= 25 and tok:                       # username sengaja tidak dibawa
                reviews.append({"rs": nama, "rating": rt, "text": txt, "tok": tok, "clean": str(r[c_clean]),
                                "date": _tgl(r[c_date]) if c_date else None})

    hospitals = []
    for nama, h in agg.items():
        addr = h["addr"].most_common(1)[0][0] if h["addr"] else ""
        lat = sorted(h["lat"])[len(h["lat"]) // 2] if h["lat"] else None   # median
        lon = sorted(h["lon"])[len(h["lon"]) // 2] if h["lon"] else None
        hospitals.append({"nama": nama, "alamat": addr, "kec": _kecamatan(addr), "lat": lat, "lon": lon,
                          "n": h["n"], "rating": (sum(h["rat"]) / len(h["rat"])) if h["rat"] else None})
    print(f"[rs] data RS: {len(hospitals)} RS, {df.shape[0]} ulasan, {len(reviews)} ulasan bisa dikutip", flush=True)
    return hospitals, reviews


RS_HOSPITALS, RS_REVIEWS = _load_rs_data(DATA_RS_PATH)
_N_REV = len(RS_REVIEWS)
_REV_INDEX = defaultdict(list)
for _i, _rv in enumerate(RS_REVIEWS):
    for _t in _rv["tok"]:
        _REV_INDEX[_t].append(_i)


def _centroids():
    pts = defaultdict(list)
    for h in RS_HOSPITALS:
        if h["kec"] and h["lat"] is not None:
            pts[h["kec"].lower()].append((h["lat"], h["lon"]))
    cen = {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)) for k, v in pts.items()}
    pwt = [c for k, c in cen.items() if k.startswith("purwokerto")]
    alias = {"purwokerto": (sum(a for a, _ in pwt) / len(pwt), sum(b for _, b in pwt) / len(pwt))} if pwt else {}
    return cen, alias


_KEC_CENTROID, _KOTA_ALIAS = _centroids()


def _words(text):
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split()


def _mentions(words, phrase, fuzzy=True):
    """True bila frasa ada di daftar kata; toleran salah ketik ringan (mis. "puwokerto selatan")."""
    pw = phrase.split()
    n = len(pw)
    if not pw or len(words) < n:
        return False
    target = " ".join(pw)
    for i in range(len(words) - n + 1):
        cand = " ".join(words[i:i + n])
        if cand == target:
            return True
        if fuzzy and len(target) >= 6 and difflib.SequenceMatcher(None, cand, target).ratio() >= 0.88:
            return True
    return False


def _origin_dari_teks(t):
    """Teks bebas (mis. 'Sokaraja', 'purwokerto selatan') -> (lat, lon, label) atau None."""
    words = _words(t)
    for kec in sorted(_KEC_CENTROID, key=len, reverse=True):
        k = " ".join(_words(kec))
        if kec in _KEC_AMBIGU:
            hit = _mentions(words, f"kecamatan {k}", fuzzy=False) or _mentions(words, f"kec {k}", fuzzy=False)
        else:
            hit = _mentions(words, k)
        if hit:
            return _KEC_CENTROID[kec][0], _KEC_CENTROID[kec][1], f"Kec. {kec.title()}"
    for name, c in _KOTA_ALIAS.items():
        if _mentions(words, name):
            return c[0], c[1], "pusat kota Purwokerto"
    return None


def _review_block(query, allowed):
    """Kutipan ulasan paling relevan dengan query (pencarian kata kunci pada clean_text)."""
    if not _N_REV:
        return ""
    toks = {t for t in preprocess(query).split()
            if len(t) >= 3 and t not in _STOP and t in _REV_INDEX and len(_REV_INDEX[t]) <= 0.4 * _N_REV}
    if not toks:
        return ""
    score = Counter()
    for t in toks:
        w = math.log(1 + _N_REV / len(_REV_INDEX[t]))
        for i in _REV_INDEX[t]:
            score[i] += w
    chosen, per_rs = [], Counter()
    for i, _ in sorted(score.items(), key=lambda kv: (-kv[1], len(RS_REVIEWS[kv[0]]["text"]))):
        rv = RS_REVIEWS[i]
        if (allowed is not None and rv["rs"] not in allowed) or per_rs[rv["rs"]] >= 2:
            continue
        per_rs[rv["rs"]] += 1
        txt = rv["text"] if len(rv["text"]) <= SNIPPET_CHARS else rv["text"][:SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."
        rt = f", rating {int(rv['rating'])}/5" if rv["rating"] is not None else ""
        chosen.append(f'- {rv["rs"]}{rt}: "{txt}"')
        if len(chosen) >= MAX_REVIEW_SNIPPETS:
            break
    return "\n".join(chosen)


INFO = {}      # info Google Maps per RS (telepon, website, jam buka), dimuat dari tabel rs_info
_BASE_HOSPITALS, _BASE_REVIEWS = RS_HOSPITALS, RS_REVIEWS   # snapshot Excel (tetap)


def reload_rs_data():
    """Gabungkan data Excel dengan ulasan baru hasil agen ops (legacy=0) dan info RS dari database,
    lalu bangun ulang indeks."""
    global RS_HOSPITALS, RS_REVIEWS, _N_REV, _REV_INDEX, _KEC_CENTROID, _KOTA_ALIAS, INFO
    with ENGINE.connect() as c:
        extra = c.execute(select(REVIEWS_TBL.c.rs_name, REVIEWS_TBL.c.rating, REVIEWS_TBL.c.text,
                                 REVIEWS_TBL.c.clean, REVIEWS_TBL.c.published_at)
                          .where(REVIEWS_TBL.c.legacy == 0, REVIEWS_TBL.c.clean.is_not(None))).all()
    hosp = {h["nama"]: dict(h) for h in _BASE_HOSPITALS}
    revs = list(_BASE_REVIEWS)
    add = defaultdict(lambda: [0, 0.0, 0])      # jumlah ulasan, jumlah rating, banyak rating
    for r in extra:
        nama = (r.rs_name or "").strip()
        if nama not in hosp:
            continue
        a = add[nama]
        a[0] += 1
        rt = _to_float(r.rating)
        if rt is not None:
            a[1] += rt
            a[2] += 1
        txt = re.sub(r"\s+", " ", str(r.text or "")).strip()
        tok = set(str(r.clean).split())
        if len(txt) >= 25 and tok:
            revs.append({"rs": nama, "rating": rt, "text": txt, "tok": tok, "clean": str(r.clean),
                         "date": (r.published_at or "")[:10] or None})
    for nama, (n, ssum, nr) in add.items():
        h = hosp[nama]
        if nr:
            h["rating"] = ((h["rating"] * h["n"] + ssum) / (h["n"] + nr)) if h["rating"] is not None else ssum / nr
        h["n"] += n
    index = defaultdict(list)
    for i, rv in enumerate(revs):
        for t in rv["tok"]:
            index[t].append(i)
    RS_HOSPITALS, RS_REVIEWS, _N_REV, _REV_INDEX = list(hosp.values()), revs, len(revs), index
    _KEC_CENTROID, _KOTA_ALIAS = _centroids()

    try:                                         # info Google Maps; boleh kosong bila belum pernah tersimpan
        with ENGINE.connect() as c:
            rows_info = c.execute(select(rs_info)).mappings().all()
        INFO = {r["rs_name"]: {"phone": r["phone"], "website": r["website"],
                               "categories": json.loads(r["categories"] or "[]"),
                               "hours": json.loads(r["hours"] or "[]"), "updated": r["updated_at"]} for r in rows_info}
    except Exception as e:
        print(f"[rs] info RS tidak dimuat: {e}", flush=True)
        INFO = {}

    print(f"[rs] data digabung: {len(RS_HOSPITALS)} RS, {len(RS_REVIEWS)} ulasan ({len(extra)} dari database), "
          f"info RS: {len(INFO)}", flush=True)


async def _run_dan_muat_ulang(*args):
    await run_routine(*args)
    try:
        await asyncio.to_thread(reload_rs_data)
    except Exception as e:
        print(f"[rs] muat ulang data gagal: {e}", flush=True)


try:
    reload_rs_data()
except Exception as e:
    print(f"[rs] gagal memuat ulasan dari database: {e}", flush=True)


# ============================================================
# Endpoint untuk tool agent Copilot Studio (hanya baca, publik, tanpa nama pengguna)
# ============================================================
OSRM_URL = os.environ.get("OSRM_URL", "https://router.project-osrm.org")   # server demo publik; ganti bila punya sendiri


def _rute_osrm(asal, tujuan):
    """(jarak_km, menit) rute MOBIL dari OSRM; menit = waktu tempuh tanpa kemacetan. None bila gagal/timeout."""
    try:
        import httpx
        r = httpx.get(f"{OSRM_URL}/route/v1/driving/{asal[1]},{asal[0]};{tujuan[1]},{tujuan[0]}",
                      params={"overview": "false"}, timeout=6)
        d = r.json()
        if r.status_code == 200 and d.get("code") == "Ok" and d.get("routes"):
            rt = d["routes"][0]
            return round(rt["distance"] / 1000, 1), max(1, round(rt["duration"] / 60))
    except Exception as e:
        print(f"[rute] OSRM gagal: {e}", flush=True)
    return None


@app.get("/rs/terdekat")
def rs_terdekat(daerah: str = "", lat: Optional[float] = None, lng: Optional[float] = None,
                batas: int = 5, rute: bool = True):
    """RS terdekat dari kecamatan (mis. Sokaraja) atau koordinat. Jarak garis lurus; untuk 5 RS pertama ditambah
    jarak dan waktu rute mobil bila tersedia."""
    origin = _origin_dari_teks(daerah) if daerah else None
    if not origin and lat is not None and lng is not None and -90 <= lat <= 90 and -180 <= lng <= 180:
        origin = (lat, lng, "lokasi pengguna")
    if not origin:
        raise HTTPException(status_code=400, detail="Sebutkan kecamatan (daerah) atau kirim lat dan lng.")
    ranked = sorted(((_haversine(origin[0], origin[1], h["lat"], h["lon"]), h)
                     for h in RS_HOSPITALS if h["lat"] is not None), key=lambda x: x[0])
    ranked = ranked[:max(1, min(batas, 10))]
    peringatan = None
    if ranked and ranked[0][0] > OUT_OF_AREA_KM:
        peringatan = ("Titik asal kemungkinan di luar Kabupaten Banyumas. Data hanya mencakup RS di Kabupaten Banyumas.")
    rutes = [None] * len(ranked)
    if rute and ranked:
        n = min(5, len(ranked))
        with ThreadPoolExecutor(max_workers=n) as ex:
            rutes[:n] = list(ex.map(lambda x: _rute_osrm(origin, (x[1]["lat"], x[1]["lon"])), ranked[:n]))
    return {
        "titik_acuan": origin[2],
        "catatan": "jarak_km = garis lurus (perkiraan). jarak_jalan_km dan perkiraan_menit = rute mobil tanpa kemacetan; "
                   "kosong bila rute tidak tersedia.",
        "peringatan": peringatan,
        "hasil": [{"nama": h["nama"], "alamat": h["alamat"], "kecamatan": h["kec"], "jarak_km": round(d, 1),
                   "jarak_jalan_km": rt[0] if rt else None, "perkiraan_menit": rt[1] if rt else None,
                   "lebih_dari_15_km": d > NEAR_KM,
                   "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                   "jumlah_ulasan": h["n"]} for (d, h), rt in zip(ranked, rutes)],
    }


@app.get("/rs/ulasan")
def rs_ulasan(topik: str, rs: str = ""):
    """Kutipan ulasan pasien yang relevan dengan topik (mis. IGD, antrean farmasi). rs opsional (sebagian nama)."""
    allowed = {h["nama"] for h in RS_HOSPITALS if rs.lower() in h["nama"].lower()} if rs else None
    return {"kutipan": _review_block(topik, allowed)}


@app.get("/rs/profil")
def rs_profil(nama: str):
    """Profil satu RS: alamat, rating, jumlah ulasan, dan kontak/jam buka bila tersimpan."""
    cocok = [h for h in RS_HOSPITALS if nama.lower() in h["nama"].lower()]
    if not cocok:
        raise HTTPException(status_code=404, detail="RS tidak ditemukan di data.")
    hasil = []
    for h in cocok[:3]:
        i = INFO.get(h["nama"], {})
        hasil.append({"nama": h["nama"], "alamat": h["alamat"], "kecamatan": h["kec"],
                      "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                      "jumlah_ulasan": h["n"], "telepon": i.get("phone"), "website": i.get("website"),
                      "jam_buka": i.get("hours") or []})
    return {"hasil": hasil}


def _deploy_model(pipe, le, metrics):
    global pipeline, label_encoder, ACCURACY, PRECISION_MACRO, RECALL_MACRO, F1_MACRO, CV_F1_MACRO
    pipeline, label_encoder = pipe, le
    ACCURACY, PRECISION_MACRO = metrics["accuracy"], metrics["precision_macro"]
    RECALL_MACRO, F1_MACRO, CV_F1_MACRO = metrics["recall_macro"], metrics["f1_macro"], metrics["cv_f1_macro"]


def _current_model():
    return pipeline, label_encoder, {"accuracy": ACCURACY, "precision_macro": PRECISION_MACRO,
                                     "recall_macro": RECALL_MACRO, "f1_macro": F1_MACRO,
                                     "cv_f1_macro": CV_F1_MACRO}


# ============================================================
# Pipeline data (ambil ulasan baru, klasifikasi, retrain). Rutinitas biasa, TIDAK terkait chatbot.
# ============================================================
OPS = OpsTools(ENGINE, preprocess, _current_model, _deploy_model)
AGENT_RUN_TOKEN = os.environ.get("AGENT_RUN_TOKEN")


def _need_token(tok):
    if not AGENT_RUN_TOKEN or not hmac.compare_digest((tok or "").encode(), AGENT_RUN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="token salah")

_retrain = {"berjalan": False, "hasil": None}


def _retrain_job():
    try:
        _retrain["hasil"] = OPS.call("retrain_model", {})
    finally:
        _retrain["berjalan"] = False

@app.post("/agent/run", status_code=202)
async def agent_run(background: BackgroundTasks, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    if ops_running():
        raise HTTPException(status_code=409, detail="rutinitas sedang berjalan")
    if _retrain["berjalan"]:
        raise HTTPException(status_code=409, detail="retrain sedang berjalan")
    background.add_task(_run_dan_muat_ulang, OPS)
    return {"status": "dimulai"}

@app.get("/agent/status")
def agent_status(x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    with ENGINE.connect() as c:
        runs = c.execute(select(agent_runs).order_by(agent_runs.c.started_at.desc()).limit(5)).mappings().all()
    return {"berjalan": ops_running(), "run_terakhir": [dict(r) for r in runs], "data": OPS.status_data()}

@app.get("/agent/ringkas")
def agent_ringkas(x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    return {"siklus_berjalan": ops_running(), "retrain_berjalan": _retrain["berjalan"],
            "run_terakhir": last_run_info(ENGINE), "data": OPS.status_data(),
            "hasil_retrain_terakhir": _retrain["hasil"]}

@app.post("/agent/retrain", status_code=202)
def agent_retrain(background: BackgroundTasks, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    if ops_running() or _retrain["berjalan"]:
        raise HTTPException(status_code=409, detail="rutinitas atau retrain sedang berjalan")
    _retrain["berjalan"] = True
    _retrain["hasil"] = None
    background.add_task(_retrain_job)
    return {"status": "dimulai"}

@app.get("/health/agent")          # publik dan minimal; dipantau cron-job.org (503 = agen tidak berjalan)
def health_agent():
    s = OPS.status_kesehatan(float(os.environ.get("HEALTH_MAX_JAM", "13")))
    return JSONResponse(s, status_code=200 if s["ok"] else 503)


@app.get("/reviews.json")          # sumber data dashboard (publik, tanpa nama pengguna)
def reviews_json():
    return export_reviews(ENGINE)


@app.get("/agent/ragu")            # antrean ulasan yang confidence-nya rendah, untuk dicek manusia bila mau
def agent_ragu(batas: int = 20, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    return OPS.lihat_ulasan_ragu(batas)


class KoreksiInput(BaseModel):
    review_id: str
    label: str


@app.post("/koreksi")              # koreksi manusia; selalu menang atas label agen
def koreksi(data: KoreksiInput, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    r = OPS.simpan_koreksi_manusia(data.review_id, data.label)
    if "error" in r:
        raise HTTPException(status_code=400, detail=r["error"])
    return r


# ============================================================
# Untuk agen NOTIFIKASI Copilot Studio (bertoken: header x-agent-token = AGENT_RUN_TOKEN).
# Alur: ambil ulasan negatif baru -> agen mengirim email lewat Outlook -> tandai terkirim.
# Aktifkan dengan NOTIFY_MODE=copilot di Render supaya server berhenti mengirim email sendiri.
# ============================================================
@app.get("/ulasan/negatif-baru")
def ulasan_negatif_baru(batas: int = 20, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    return OPS.negatif_baru(batas)

@app.get("/ulasan/email-html")
def ulasan_email_html(x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    
    # Ambil ulasan negatif yang belum dikirim (maksimal 50)
    rows = OPS._pending_negative(50)
    
    if not rows:
        return {"subjek": "Tidak ada ulasan negatif baru", "html": "<p>Tidak ada ulasan negatif baru saat ini.</p>"}
        
    # Panggil fungsi _susun_email untuk membuat HTML cantik
    subjek, teks, html = _susun_email(rows)
    
    # Kembalikan dalam format JSON
    return {"subjek": subjek, "html": html}

class TandaiInput(BaseModel):
    review_ids: list[str]

import httpx
import asyncio

@app.get("/ulasan/copilot-email")
async def ulasan_copilot_email(x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    
    # 1. Ambil 1 ulasan negatif
    rows = OPS._pending_negative(1)
    if not rows:
        return {"subjek": "Tidak ada ulasan", "html": "<p>Tidak ada ulasan negatif.</p>"}
    
    ulasan = rows[0]
    teks_ulasan = f"RS: {ulasan.rs_name}. Rating: {ulasan.rating}. Ulasan: {ulasan.text}"
    
    # 2. Coba panggil Copilot Studio
    try:
        DIRECT_LINE_SECRET = "https://default90affe0fc2a34108bb986ceb4e94ef.15.environment.api.powerplatform.com/powervirtualagents/botsbyschema/new_AnalisisGeosentimen/directline/token?api-version=2022-03-01-preview" # <--- WAJIB GANTI DENGAN SECRET ASLI
        headers = {"Authorization": f"Bearer {DIRECT_LINE_SECRET}"}
        
        async with httpx.AsyncClient() as client:
            # Buka percakapan
            res = await client.post("https://directline.botframework.com/v3/directline/conversations", headers=headers, timeout=10)
            if res.status_code != 200:
                raise Exception("Direct Line gagal")
                
            conv_id = res.json().get("conversationId")
            
            # Kirim pesan
            await client.post(f"https://directline.botframework.com/v3/directline/conversations/{conv_id}/activities", 
                              headers=headers, 
                              json={"type": "message", "from": {"id": "user1"}, "text": teks_ulasan}, timeout=10)
            
            # Tunggu agen berpikir
            await asyncio.sleep(5) 
            
            # Ambil balasan
            res_act = await client.get(f"https://directline.botframework.com/v3/directline/conversations/{conv_id}/activities", headers=headers, timeout=10)
            activities = res_act.json().get("activities", [])
            
            balasan_agen = None
            for act in activities:
                if act.get("from", {}).get("id") != "user1" and act.get("type") == "message":
                    balasan_agen = act.get("text")
                    break
            
            # Kalau agen tidak merespons atau kena limit
            if not balasan_agen or "Batas penggunaan" in balasan_agen:
                raise Exception("Kuota Copilot habis atau tidak merespons")

            subjek = f"[Geosentimen] Ulasan Negatif dari {ulasan.rs_name}"
            html = balasan_agen.replace("\n", "<br>")
            return {"subjek": subjek, "html": html}

    except Exception as e:
        # 3. MODE FALLBACK (JIKA COPILOT ERROR / LIMIT)
        print(f"[fallback] Copilot error: {e}. Menggunakan mode template HTML.")
        # Panggil fungsi _susun_email yang sudah ada di ops_agent.py
        subjek, teks, html = _susun_email([ulasan]) 
        return {"subjek": subjek, "html": html}

@app.post("/ulasan/tandai-terkirim")
def ulasan_tandai_terkirim(data: TandaiInput, x_agent_token: str = Header(default=None)):
    _need_token(x_agent_token)
    return OPS.tandai_terkirim(data.review_ids)


# ============================================================
# Untuk agen chat Copilot Studio (publik, baca saja, tanpa nama pengguna; datanya sama dengan /reviews.json)
# ============================================================
@app.get("/rs/ringkasan")
def rs_ringkasan(topik: str = "", rs: str = "", batas: int = 8):
    """Per RS: rating, jumlah ulasan, sebaran sentimen; bila topik diisi (mis. IGD) ditambah ulasan yang menyebutnya."""
    return OPS.ringkasan_rs(topik, rs, batas)


@app.get("/ulasan/per-jam")
def ulasan_per_jam(jam: int, rs: str = "", batas: int = 8):
    """Jumlah dan isi ulasan yang ditulis pada jam tertentu (WIB) beserta label sentimennya."""
    r = OPS.ulasan_per_jam(jam, rs, batas)
    if "error" in r:
        raise HTTPException(status_code=400, detail=r["error"])
    return r


# ============================================================
# Endpoint /refresh -- tombol "Ambil Ulasan Terbaru" di dashboard menjalankan rutinitas ambil ulasan.
# Publik, jadi dibatasi jeda antar-run supaya kredit Apify tidak habis oleh klik berulang.
# ============================================================
REFRESH_COOLDOWN_SEC = int(os.environ.get("REFRESH_COOLDOWN_SEC", "3600"))
_last_refresh = 0.0


@app.post("/refresh")
async def refresh(background: BackgroundTasks):
    global _last_refresh
    if ops_running():
        raise HTTPException(status_code=409, detail="Update sedang berjalan. Tunggu beberapa menit.")
    sisa = REFRESH_COOLDOWN_SEC - (time.time() - _last_refresh)
    if sisa > 0:
        raise HTTPException(status_code=429,
                            detail=f"Update baru saja dijalankan. Coba lagi sekitar {int(sisa // 60) + 1} menit lagi.")
    _last_refresh = time.time()
    background.add_task(_run_dan_muat_ulang, OPS)
    return {"status": "dimulai"}


@app.get("/refresh/status")
def refresh_status():
    return last_run_info(ENGINE)


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