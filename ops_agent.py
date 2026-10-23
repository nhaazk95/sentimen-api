import asyncio
import hashlib
import html as _html
import io
import json
import math
import os
import re
import smtplib
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.message import EmailMessage

import joblib
import numpy as np
from sqlalchemy import (Column, Float, Integer, LargeBinary, MetaData, String, Table, Text,
                        bindparam, create_engine, func, insert, select, update)

# ------------------------------------------------------------------ catatan versi ini
# - Tidak ada LLM/Groq lagi. Rutinitas = langkah tetap di kode (ambil ulasan -> label SVM -> email -> retrain).
# - Email ulasan negatif punya dua mode (env NOTIFY_MODE):
#     "kode"    (bawaan) : dikirim server lewat Resend/Brevo/SMTP seperti sebelumnya
#     "copilot"          : server TIDAK mengirim; agen Copilot Studio mengambil lewat negatif_baru() lalu
#                          mengirim lewat Outlook dan memanggil tandai_terkirim()
# - Tidak lagi bergantung pada agent.py.

# ------------------------------------------------------------------ konfigurasi
PLACES_FILE = os.environ.get("PLACES_FILE", "places.json")          # daftar URL Google Maps tiap RS
APIFY_ACTOR = os.environ.get("APIFY_ACTOR", "compass/crawler-google-places")   # sama dengan scrape_update.py
APIFY_MAX_CHARGE_USD = os.environ.get("APIFY_MAX_CHARGE_USD", "2")            # batas biaya per run
DEFAULT_MAX_REVIEWS = 5        # per RS per run, sama dengan MAX_REVIEWS_PER_RS di scrape_update.py
NOTIFY_MAX_AGE_DAYS = 14       # email hanya untuk ulasan yang ditulis dalam N hari terakhir
INDONESIA_BBOX = {"lat_min": -11, "lat_max": 6, "lng_min": 95, "lng_max": 141}
TRAIN_POOL = os.environ.get("TRAIN_POOL", "data/training/train_pool.csv")
TEST_SET = os.environ.get("TEST_SET", "data/training/test_set.csv")   # label manusia, TIDAK pernah ikut dilatih
TEXT_COL = os.environ.get("TEXT_COL", "clean_text")      # kolom teks yang sudah dibersihkan (dipakai bila ada)
RAW_TEXT_COL = os.environ.get("RAW_TEXT_COL", "text")    # kolom teks mentah (dibersihkan dengan preprocess saat retrain)
LABEL_COL = os.environ.get("LABEL_COL", "label")
MIN_NEW_CORRECTIONS = int(os.environ.get("MIN_NEW_CORRECTIONS", "30"))
MIN_F1_GAIN = float(os.environ.get("MIN_F1_GAIN", "0.0"))
MAX_NEG_RECALL_DROP = 0.05    # recall kelas negatif tidak boleh turun lebih dari ini (alert email bergantung padanya)
ALERT_COOLDOWN_H = 12          # satu jenis peringatan maksimal sekali per 12 jam
RETRAIN_OLEH_COPILOT = (os.environ.get("RETRAIN_MODE") or "kode").strip().lower() == "copilot"
# ------------------------------------------------------------------ database
md = MetaData()
reviews = Table(
    "reviews", md,
    Column("review_id", String(255), primary_key=True),
    Column("rs_name", String(255)), Column("place_id", String(255)),
    Column("rating", Float), Column("text", Text), Column("clean", Text),
    Column("label", String(20)), Column("confidence", Float),
    Column("corrected_label", String(20)), Column("label_source", String(10)),   # 'manusia' (nilai 'agen' hanya sisa versi lama)
    Column("published_at", String(40)), Column("scraped_at", String(40)),
    Column("notified", Integer, nullable=False, default=0),
    Column("username", String(255)), Column("address", Text), Column("lat", Float), Column("lon", Float),
    Column("processed_at", String(40)),
    Column("legacy", Integer, nullable=False, default=0),   # 1 = diimpor dari reviews.json lama (id-nya bukan reviewId Google)
)
models = Table("models", md, Column("version", String(40), primary_key=True), Column("created_at", String(40)),
               Column("blob", LargeBinary), Column("metrics", Text), Column("active", Integer, default=0))
agent_runs = Table("agent_runs", md, Column("run_id", String(40), primary_key=True), Column("started_at", String(40)),
                   Column("finished_at", String(40)), Column("status", String(20)), Column("report", Text),
                   Column("log", Text))
kv = Table("kv", md, Column("k", String(80), primary_key=True), Column("v", Text))
rs_info = Table("rs_info", md, Column("rs_name", String(255), primary_key=True), Column("phone", String(60)),
                Column("website", String(300)), Column("categories", Text), Column("hours", Text),
                Column("extra", Text), Column("updated_at", String(40)))

notif_log = Table("notif_log", md,
                  Column("id", String(40), primary_key=True),
                  Column("event", String(10)),        # 'served' = diambil agen, 'marked' = ditandai terkirim
                  Column("review_id", String(255)),
                  Column("ts", String(40)))

def make_engine(url=None):
    url = url or os.environ.get("DATABASE_URL") or "sqlite:///agent.db"
    for old in ("postgres://", "postgresql://"):
        if url.startswith(old):
            url = "postgresql+psycopg2://" + url[len(old):]
    return create_engine(url, pool_pre_ping=True)


def init_db(engine):
    md.create_all(engine)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp(v, lo, hi, default):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _num(v):
    """NaN/Infinity tidak valid di JSON dan membuat respons 500; ganti dengan None."""
    return None if isinstance(v, float) and not math.isfinite(v) else v


def _notify_via_copilot():
    return (os.environ.get("NOTIFY_MODE") or "kode").strip().lower() == "copilot"


# ------------------------------------------------------------------ kata kunci topik ulasan
# Sama dengan TOPIK/kataKunci di index.html lama (dipakai agent lama untuk "ULASAN YANG MENYEBUT TOPIK per RS").
_TOPIK = {
    "persalinan": ["persalinan", "lahiran", "melahirkan", "bersalin", "bidan", "caesar", "sesar", "sc",
                   "kandungan", "obgyn", "obsgyn", "hamil", "kehamilan", "antenatal", "prenatal", "usg", "spog"],
    "igd": ["igd", "ugd", "darurat"],
    "farmasi": ["farmasi", "obat", "apotek", "resep"],
    "antre": ["antre", "antri", "antrian", "tunggu"],
    "bpjs": ["bpjs", "jkn", "kis"],
    "gigi": ["gigi"],
    "anak": ["anak", "bayi", "nicu", "balita"],
    "bedah": ["operasi", "bedah"],
    "radiologi": ["radiologi", "rontgen", "usg", "ct", "mri"],
    "parkir": ["parkir"],
    "kebersihan": ["bersih", "kotor", "bau"],
}
_STOP_TOPIK = {"yang", "dan", "di", "ke", "dari", "untuk", "rs", "rumah", "sakit", "apa", "ada", "yg", "mana",
               "bagaimana", "dengan", "atau", "saya", "mau", "cari", "bisa", "tolong", "dong", "ya", "itu", "ini",
               "terbaik", "dimana", "bagus", "baik", "jelek", "banyak"}


def _kata_kunci(pertanyaan):
    kata = [w for w in re.split(r"[^a-z0-9]+", str(pertanyaan or "").lower()) if len(w) >= 3 and w not in _STOP_TOPIK]
    hasil = []
    for topik, sinonim in _TOPIK.items():
        if topik in kata or any(w == x or (len(x) >= 5 and x in w) for w in kata for x in sinonim):
            hasil += [x for x in sinonim if x not in hasil]
    return hasil or list(dict.fromkeys(kata))


def _pola_topik(kunci):
    if not kunci:
        return None
    pola = [re.escape(k) if len(k) >= 5 else r"\b" + re.escape(k) + r"\b" for k in kunci]
    return re.compile("(" + "|".join(pola) + ")", re.I)


# ------------------------------------------------------------------ penyimpanan model
def save_model(engine, pipeline, label_encoder, metrics):
    buf = io.BytesIO()
    joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder}, buf)
    version = "v" + time.strftime("%Y%m%d%H%M%S")
    with engine.begin() as c:
        c.execute(update(models).values(active=0))
        c.execute(insert(models).values(version=version, created_at=_now(), blob=buf.getvalue(),
                                        metrics=json.dumps(metrics), active=1))
    return version


def load_active_model(engine):
    """(pipeline, label_encoder, metrics) dari database, atau None bila belum pernah ada retrain."""
    with engine.connect() as c:
        row = c.execute(select(models.c.blob, models.c.metrics).where(models.c.active == 1)
                        .order_by(models.c.created_at.desc())).first()
    if not row:
        return None
    obj = joblib.load(io.BytesIO(row.blob))
    return obj["pipeline"], obj["label_encoder"], json.loads(row.metrics)


def last_run_info(engine):
    """Ringkasan run terakhir untuk dashboard (publik, tanpa log atau laporan). Bentuknya meniru status
    workflow GitHub agar JavaScript dashboard hampir tidak berubah."""
    with engine.connect() as c:
        r = c.execute(select(agent_runs.c.status, agent_runs.c.started_at)
                      .order_by(agent_runs.c.started_at.desc()).limit(1)).first()
        ok = c.execute(select(agent_runs.c.finished_at).where(agent_runs.c.status == "ok")
                       .order_by(agent_runs.c.finished_at.desc()).limit(1)).scalar()
    if not r:
        return {"status": "unknown", "conclusion": None, "created_at": None, "last_run_utc": ok}
    return {"status": "in_progress" if r.status == "running" else "completed",
            "conclusion": None if r.status == "running" else ("success" if r.status == "ok" else "failure"),
            "created_at": r.started_at, "last_run_utc": ok}


def export_reviews(engine):
    """Isi GET /reviews.json: kolom dan namanya sama persis dengan docs/data/reviews.json yang dibaca dashboard."""
    with engine.connect() as c:
        rows = c.execute(select(reviews).where(reviews.c.label.is_not(None))
                         .order_by(reviews.c.published_at.desc())).mappings().all()
    return [{"Nama RS": r["rs_name"], "Username": r["username"], "Rating": _num(r["rating"]),
             "Waktu Ulasan": r["published_at"], "Lokasi Tempat": r["address"],
             "Latitude": _num(r["lat"]), "Longitude": _num(r["lon"]), "Isi Ulasan": r["text"], "id": r["review_id"],
             "Sentimen_Prediksi": r["corrected_label"] or r["label"], "Confidence": _num(r["confidence"]),
             "Processed_At": r["processed_at"], "Teks_Bersih": r["clean"],
             "Label_Manual": r["corrected_label"] if r["label_source"] == "manusia" else None,
             "Notified": bool(r["notified"])} for r in rows]


# ------------------------------------------------------------------ alat
def _get_field(obj, key):
    """Ambil field dari hasil Apify, baik berupa dict maupun object (beda versi apify-client)."""
    if isinstance(obj, dict):
        return obj.get(key)
    snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in key).lstrip("_")
    return getattr(obj, snake, None) if hasattr(obj, snake) else getattr(obj, key, None)


def _apify_fetch(token, run_input):
    """Mengembalikan daftar 'place' (tiap place berisi daftar 'reviews'), sama seperti scrape_update.py."""
    from apify_client import ApifyClient
    client = ApifyClient(token)
    run = client.actor(APIFY_ACTOR).call(run_input=run_input, max_total_charge_usd=Decimal(APIFY_MAX_CHARGE_USD))
    return list(client.dataset(_get_field(run, "defaultDatasetId")).iterate_items())


def _smtp_send(user, password, recipients, msg):
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(user, password)
        s.send_message(msg, from_addr=user, to_addrs=recipients)


def _http_send(provider, recipients, subject, text, html=None):
    """Kirim email lewat API HTTPS (port 443). Render gratis memblokir port SMTP 25/465/587 sejak 26 Sep 2025."""
    import httpx
    if provider == "resend":
        r = httpx.post("https://api.resend.com/emails", timeout=30,
                       headers={"Authorization": f"Bearer {os.environ.get('RESEND_API_KEY')}"},
                       json={"from": os.environ.get("EMAIL_FROM") or "Geosentimen <onboarding@resend.dev>",
                             "to": recipients, "subject": subject, "text": text,
                             **({"html": html} if html else {})})
    elif provider == "brevo":
        r = httpx.post("https://api.brevo.com/v3/smtp/email", timeout=30,
                       headers={"api-key": os.environ.get("BREVO_API_KEY", ""), "accept": "application/json"},
                       json={"sender": {"email": os.environ.get("EMAIL_FROM"), "name": "Geosentimen RS Banyumas"},
                             "to": [{"email": x} for x in recipients], "subject": subject, "textContent": text,
                             **({"htmlContent": html} if html else {})})
    else:
        raise ValueError(f"provider email tidak dikenal: {provider}")
    if r.status_code >= 300:
        raise RuntimeError(f"{provider} HTTP {r.status_code}: {r.text[:200]}")


def _email_provider():
    p = (os.environ.get("EMAIL_PROVIDER") or "").strip().lower()
    if p:
        return p
    if os.environ.get("RESEND_API_KEY"):
        return "resend"
    if os.environ.get("BREVO_API_KEY"):
        return "brevo"
    return "smtp"


BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
         "November", "Desember"]


def _wib(iso=None):
    """ISO UTC (atau sekarang) -> datetime WIB (UTC+7) tanpa zona; None bila tidak terbaca."""
    try:
        d = datetime.strptime(str(iso)[:19], "%Y-%m-%dT%H:%M:%S") if iso else datetime.utcnow()
    except ValueError:
        return None
    return d + timedelta(hours=7)


def _tgl(iso):
    d = _wib(iso)
    return f"{d.day} {BULAN[d.month - 1]} {d.year}" if d else "-"


def _bintang(rating):
    try:
        n = max(0, min(5, int(round(float(rating)))))
    except (TypeError, ValueError):
        return "-", "-"
    return "★" * n + "☆" * (5 - n), f"{n}/5"


def _potong(text, n=700):
    t = str(text or "").replace("\r", "").strip()
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    if len(t) <= n:
        return t
    return t[:n].rsplit(" ", 1)[0] + " ..."


def _susun_email(rows, ringkasan=""):
    """Kembalikan (subjek, teks, html) untuk email ulasan negatif (mode NOTIFY_MODE=kode).
    Teks ulasan di-escape: itu data pihak luar."""
    esc = _html.escape
    by = {}
    for r in sorted(rows, key=lambda r: str(r.published_at or ""), reverse=True):
        by.setdefault(r.rs_name or "RS tidak diketahui", []).append(r)
    n, k = len(rows), len(by)
    now = _wib()
    waktu = f"{now.day} {BULAN[now.month - 1]} {now.year}, {now.hour:02d}.{now.minute:02d} WIB"
    ringkasan = " ".join((ringkasan or "").split())[:800]
    subject = f"[Geosentimen] {n} ulasan negatif baru di {k} rumah sakit"

    # ---- teks biasa
    t = [f"ULASAN NEGATIF BARU - Dashboard Sentimen RS Banyumas", waktu,
         f"{n} ulasan negatif di {k} rumah sakit", ""]
    if ringkasan:
        t += ["RINGKASAN", ringkasan, ""]
    for rs, items in by.items():
        t.append(f"== {rs} ({len(items)} ulasan) ==")
        for r in items:
            t += [f"[{_bintang(r.rating)[1]}] {_tgl(r.published_at)}", _potong(r.text), ""]
    t.append("Catatan: prediksi sentimen dibuat otomatis oleh model SVM dan dapat keliru. Mohon periksa isi ulasan.")
    teks = "\n".join(t)

    # ---- HTML (tabel + gaya inline agar tampil konsisten di Gmail/Outlook)
    def kartu(r):
        bintang, skor = _bintang(r.rating)
        isi = esc(_potong(r.text)).replace("\n", "<br>")
        return (
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:10px 0;'
            'background:#fff7f7;border:1px solid #fecaca;border-left:4px solid #dc2626;border-radius:6px;">'
            '<tr><td style="padding:12px 14px;font-family:Arial,Helvetica,sans-serif;">'
            f'<div style="font-size:13px;font-weight:700;color:#b91c1c;letter-spacing:1px;">{bintang}'
            f'<span style="font-weight:400;letter-spacing:0;color:#6b7280;">&nbsp; {skor} &bull; {esc(_tgl(r.published_at))}</span></div>'
            f'<div style="font-size:14px;line-height:1.6;color:#1f2937;margin-top:8px;">{isi}</div>'
            '</td></tr></table>')

    bagian = []
    for rs, items in by.items():
        bagian.append(
            '<tr><td style="padding:20px 28px 0 28px;font-family:Arial,Helvetica,sans-serif;">'
            f'<div style="font-size:16px;font-weight:700;color:#111827;">{esc(rs)}</div>'
            f'<div style="font-size:12px;color:#6b7280;margin-top:2px;">{len(items)} ulasan negatif</div></td></tr>'
            f'<tr><td style="padding:2px 28px 0 28px;">{"".join(kartu(r) for r in items)}</td></tr>')
    box = ""
    if ringkasan:
        box = ('<tr><td style="padding:16px 28px 0 28px;font-family:Arial,Helvetica,sans-serif;">'
               '<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:12px 14px;">'
               '<div style="font-size:12px;font-weight:700;color:#92400e;letter-spacing:.5px;">RINGKASAN KELUHAN</div>'
               f'<div style="font-size:14px;line-height:1.6;color:#78350f;margin-top:4px;">{esc(ringkasan)}</div>'
               '</div></td></tr>')
    html = (
        '<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
        '<body style="margin:0;padding:0;background:#f3f4f6;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;">'
        '<tr><td align="center" style="padding:24px 12px;">'
        '<table role="presentation" width="600" cellpadding="0" cellspacing="0" '
        'style="width:100%;max-width:600px;background:#ffffff;border-radius:10px;overflow:hidden;">'
        '<tr><td style="background:#1e3a8a;padding:22px 28px;font-family:Arial,Helvetica,sans-serif;">'
        '<div style="font-size:12px;color:#bfdbfe;letter-spacing:1px;">DASHBOARD SENTIMEN RS BANYUMAS</div>'
        '<div style="font-size:22px;font-weight:700;color:#ffffff;margin-top:4px;">Ulasan Negatif Baru</div>'
        f'<div style="font-size:13px;color:#dbeafe;margin-top:6px;">{esc(waktu)}</div></td></tr>'
        '<tr><td style="padding:18px 28px 0 28px;font-family:Arial,Helvetica,sans-serif;">'
        f'<span style="font-size:30px;font-weight:700;color:#dc2626;">{n}</span>'
        f'<span style="font-size:14px;color:#374151;"> ulasan negatif di {k} rumah sakit</span></td></tr>'
        + box + "".join(bagian) +
        '<tr><td style="padding:22px 28px 24px 28px;font-family:Arial,Helvetica,sans-serif;">'
        '<div style="border-top:1px solid #e5e7eb;padding-top:14px;font-size:12px;line-height:1.6;color:#6b7280;">'
        'Prediksi sentimen dibuat otomatis oleh model SVM dan dapat keliru. Mohon periksa isi ulasan sebelum menindaklanjuti.<br>'
        'Email ini dikirim otomatis oleh sistem Geosentimen.</div></td></tr>'
        '</table></td></tr></table></body></html>')
    return subject, teks, html


class OpsTools:
    def __init__(self, engine, preprocess, get_model, deploy, fetch_items=None, send_mail=None, http_send=None):
        """get_model() -> (pipeline, label_encoder, metrics); deploy(pipeline, le, metrics) mengganti model aktif."""
        self.engine = engine
        self.preprocess = preprocess
        self.get_model = get_model
        self.deploy = deploy
        self.fetch_items = fetch_items or _apify_fetch
        self.send_mail = send_mail or _smtp_send
        self.http_send = http_send or _http_send

    # ---- util
    def _kv_get(self, k, default=None):
        with self.engine.connect() as c:
            v = c.execute(select(kv.c.v).where(kv.c.k == k)).scalar()
        return default if v is None else v

    def _kv_set(self, k, v):
        with self.engine.begin() as c:
            if not c.execute(update(kv).where(kv.c.k == k).values(v=str(v))).rowcount:
                c.execute(insert(kv).values(k=k, v=str(v)))

    def _canon(self, label):
        """Samakan huruf besar/kecil label dengan kelas pada label_encoder aktif; None bila bukan kelas valid."""
        _, le, _ = self.get_model()
        m = {str(c).lower(): str(c) for c in le.classes_}
        return m.get(str(label).strip().lower())

    @staticmethod
    def _eff():
        return func.lower(func.coalesce(reviews.c.corrected_label, reviews.c.label))

    def _place_urls(self):
        if not os.path.exists(PLACES_FILE):
            return []
        with open(PLACES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        urls = [(x if isinstance(x, str) else (x or {}).get("url")) for x in data]
        return [u for u in urls if u]

    def _expire_old_pending(self):
        """Ulasan yang ditulis lebih dari NOTIFY_MAX_AGE_DAYS hari lalu tidak diemailkan (seperti scrape_update.py)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=NOTIFY_MAX_AGE_DAYS)).strftime("%Y-%m-%dT%H:%M:%S")
        with self.engine.begin() as c:
            c.execute(update(reviews).where(
                reviews.c.notified == 0,
                (reviews.c.published_at.is_(None)) | (reviews.c.published_at < cutoff)).values(notified=1))

    def _since_date(self):
        """Hanya ambil ulasan sejak 7 hari sebelum ulasan terakhir di database (hemat kredit Apify)."""
        with self.engine.connect() as c:
            mx = c.execute(select(func.max(reviews.c.published_at))).scalar()
        if not mx:
            return None
        try:
            return (datetime.fromisoformat(str(mx)[:10]) - timedelta(days=7)).strftime("%Y-%m-%d")
        except ValueError:
            return None

    def _simpan_info(self, places):
        """Simpan telepon, website, kategori, dan jam buka tiap RS dari hasil Apify."""
        now = _now()
        with self.engine.begin() as c:
            for p in places:
                nama = (p.get("title") or "").strip()
                if not nama:
                    continue
                cats = p.get("categories") or ([p["categoryName"]] if p.get("categoryName") else [])
                row = {"phone": p.get("phone") or p.get("phoneUnformatted"), "website": p.get("website"),
                       "categories": json.dumps(cats, ensure_ascii=False),
                       "hours": json.dumps(p.get("openingHours") or [], ensure_ascii=False),
                       "extra": json.dumps(p.get("additionalInfo") or {}, ensure_ascii=False)[:20000], "updated_at": now}
                if not c.execute(update(rs_info).where(rs_info.c.rs_name == nama).values(**row)).rowcount:
                    c.execute(insert(rs_info).values(rs_name=nama, **row))

    # ---- 1. status
    def status_data(self):
        self._expire_old_pending()
        eff = self._eff()
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(reviews)).scalar()
            unl = c.execute(select(func.count()).select_from(reviews).where(reviews.c.label.is_(None))).scalar()
            pend = c.execute(select(func.count()).select_from(reviews)
                             .where(eff == "negatif", reviews.c.notified == 0)).scalar()
            corr = c.execute(select(func.count()).select_from(reviews)
                             .where(reviews.c.corrected_label.is_not(None))).scalar()
            by = c.execute(select(eff, func.count()).where(eff.is_not(None)).group_by(eff)).all()
        _, _, metrics = self.get_model()
        last = int(self._kv_get("last_retrain_count", 0))
        return {"total_ulasan": total, "belum_berlabel": unl, "negatif_belum_diemail": pend,
                "per_label": {k: v for k, v in by}, "koreksi_total": corr,
                "koreksi_sejak_retrain_terakhir": corr - last, "syarat_retrain": MIN_NEW_CORRECTIONS,
                "f1_macro_model_aktif": metrics.get("f1_macro"),
                "retrain_terakhir": self._kv_get("last_retrain_result", "belum pernah"),
                "email_oleh": "agen Copilot Studio" if _notify_via_copilot() else "server"}

    # ---- 2. ambil ulasan
    def ambil_ulasan_baru(self, maks_per_rs=DEFAULT_MAX_REVIEWS):
        token = os.environ.get("APIFY_TOKEN")
        if not token:
            return {"error": "APIFY_TOKEN belum diset"}
        urls = self._place_urls()
        if not urls:
            return {"error": f"daftar RS kosong; isi {PLACES_FILE} dengan URL Google Maps tiap RS"}
        maks = _clamp(maks_per_rs, 1, 100, DEFAULT_MAX_REVIEWS)     # batas pengaman kredit Apify
        run_input = {"startUrls": [{"url": u} for u in urls], "language": "id", "countryCode": "id",
                     "maxCrawledPlacesPerSearch": 1, "maxReviews": maks, "reviewsSort": "newest",
                     "maxImages": 0, "maxQuestions": 0}
        since = self._since_date()
        if since:
            run_input["reviewsStartDate"] = since
        try:
            places = self.fetch_items(token, run_input)
        except Exception as e:
            return {"error": f"Apify gagal: {e}"}
        try:
            self._simpan_info(places)
        except Exception as e:
            print(f"[ops] simpan info RS gagal: {e}", flush=True)

        cand, notext, luar, total = {}, 0, 0, 0
        bb = INDONESIA_BBOX
        for place in places:
            nama = place.get("title")
            loc = place.get("location") or {}
            lat = loc.get("lat", place.get("latitude"))
            lon = loc.get("lng", place.get("longitude"))
            for r in place.get("reviews") or []:
                total += 1
                raw = r.get("text")
                if not raw or not str(raw).strip():
                    notext += 1
                    continue
                if lat is None or lon is None or not (bb["lat_min"] <= lat <= bb["lat_max"]
                                                      and bb["lng_min"] <= lon <= bb["lng_max"]):
                    luar += 1       # actor kadang salah menemukan tempat di luar negeri
                    continue
                # Rumus id SAMA PERSIS dengan scrape_update.py, jadi ulasan lama otomatis terdeteksi
                rid = hashlib.sha256(f"{nama}|{r.get('name')}|{r.get('publishedAtDate')}|{raw}".encode("utf-8")
                                     ).hexdigest()[:16]
                cand[rid] = {"review_id": rid, "rs_name": nama, "username": r.get("name"), "rating": r.get("stars"),
                             "published_at": r.get("publishedAtDate"), "address": place.get("address"),
                             "lat": lat, "lon": lon, "text": raw, "scraped_at": _now(), "notified": 0, "legacy": 0}
        existing = set()
        with self.engine.connect() as c:
            for part in _chunks(list(cand), 500):
                existing.update(c.execute(select(reviews.c.review_id).where(reviews.c.review_id.in_(part))).scalars())
        new = [v for k, v in cand.items() if k not in existing]
        if new:
            with self.engine.begin() as c:
                c.execute(insert(reviews), new)
        labeled = None
        if new:
            try:
                labeled = self.label_ulasan(batas=5000)     # langsung dilabeli SVM
            except Exception as e:
                labeled = {"error": f"pelabelan gagal: {e}"}
        return {"dilabeli_otomatis": labeled, "ulasan_diterima": total, "baru": len(new), "sudah_ada": len(cand) - len(new),
                "tanpa_teks_dilewati": notext, "di_luar_indonesia_dibuang": luar,
                "sejak_tanggal": since, "baru_per_rs": dict(Counter(v["rs_name"] for v in new))}

    # ---- 3. label SVM
    def _predict(self, clean_texts):
        pipe, le, _ = self.get_model()
        labels = le.inverse_transform(pipe.predict(clean_texts))
        X = pipe.named_steps["tfidf"].transform(clean_texts)
        S = pipe.named_steps["svm"].decision_function(X)
        if S.ndim == 1:
            conf = np.abs(S)
        else:
            P = np.sort(S, axis=1)
            conf = P[:, -1] - P[:, -2]
        return [str(x) for x in labels], [round(float(x), 4) for x in conf]

    def label_ulasan(self, batas=500, semua=False):
        """Label ulasan yang belum berlabel (atau SEMUA ulasan bila semua=True, dipakai setelah model baru dipasang)."""
        q = select(reviews.c.review_id, reviews.c.text, reviews.c.clean)
        if not semua:
            q = q.where(reviews.c.label.is_(None))
        with self.engine.connect() as c:
            rows = c.execute(q.limit(_clamp(batas, 1, 100000, 500))).all()
        if not rows:
            return {"dilabeli": 0}
        clean = [r.clean if (semua and r.clean) else self.preprocess(r.text) for r in rows]
        labels, conf = self._predict(clean)
        stmt = (update(reviews).where(reviews.c.review_id == bindparam("b_id"))
                .values(label=bindparam("b_label"), confidence=bindparam("b_conf"), clean=bindparam("b_clean"),
                        processed_at=bindparam("b_pa")))
        pa = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        with self.engine.begin() as c:
            c.execute(stmt, [{"b_id": r.review_id, "b_label": l, "b_conf": cf, "b_clean": cl, "b_pa": pa}
                             for r, l, cf, cl in zip(rows, labels, conf, clean)])
        return {"dilabeli": len(rows), "per_label": dict(Counter(labels)),
                "confidence_rata2": round(float(np.mean(conf)), 3)}

    # ---- 4. email ulasan negatif
    def _pending_negative(self, limit):
        self._expire_old_pending()
        with self.engine.connect() as c:
            return c.execute(select(reviews.c.review_id, reviews.c.rs_name, reviews.c.rating, reviews.c.text,
                                    reviews.c.published_at)
                             .where(self._eff() == "negatif", reviews.c.notified == 0)
                             .order_by(reviews.c.scraped_at).limit(limit)).all()

    def lihat_ulasan_negatif_belum_dikirim(self, batas=12):
        rows = self._pending_negative(_clamp(batas, 1, 30, 12))
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(reviews)
                              .where(self._eff() == "negatif", reviews.c.notified == 0)).scalar()
        return {"total_menunggu": total, "ditampilkan": len(rows),
                "ulasan": [{"rs": r.rs_name, "rating": r.rating, "teks": (r.text or "")[:250]} for r in rows]}

    def kirim_email_negatif(self, ringkasan=""):
        """Mode NOTIFY_MODE=kode: server mengirim satu email berisi semua ulasan negatif yang menunggu."""
        to = (os.environ.get("NOTIFY_TO") or "").strip()
        if not to:
            return {"error": "NOTIFY_TO belum diset"}
        provider = _email_provider()
        if provider == "smtp":
            user, pw = os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_APP_PASSWORD")
            if not (user and pw):
                return {"error": "GMAIL_USER / GMAIL_APP_PASSWORD belum diset (atau pakai RESEND_API_KEY / BREVO_API_KEY)"}
        elif provider == "resend" and not os.environ.get("RESEND_API_KEY"):
            return {"error": "RESEND_API_KEY belum diset"}
        elif provider == "brevo" and not (os.environ.get("BREVO_API_KEY") and os.environ.get("EMAIL_FROM")):
            return {"error": "BREVO_API_KEY dan EMAIL_FROM (alamat pengirim terverifikasi di Brevo) belum diset"}
        rows = self._pending_negative(50)
        if not rows:
            return {"dikirim": 0, "catatan": "tidak ada ulasan negatif baru"}
        recipients = [x.strip() for x in to.split(",") if x.strip()]
        subject, body, html = _susun_email(rows, ringkasan)       # subjek dari kode, bukan dari teks ulasan
        try:
            if provider == "smtp":
                msg = EmailMessage()
                msg["Subject"], msg["From"], msg["To"] = subject, user, ", ".join(recipients)
                msg.set_content(body)
                msg.add_alternative(html, subtype="html")
                self.send_mail(user, pw, recipients, msg)
            else:
                self.http_send(provider, recipients, subject, body, html)
        except Exception as e:
            hint = ""
            if provider == "smtp":
                hint = " (Render gratis memblokir port SMTP; set RESEND_API_KEY atau BREVO_API_KEY)"
            return {"error": f"email gagal terkirim lewat {provider}: {e}{hint}; ulasan tetap antre dan dicoba lagi"}
        with self.engine.begin() as c:       # tandai SETELAH berhasil, supaya tidak ada ulasan yang terlewat
            c.execute(update(reviews).where(reviews.c.review_id.in_([r.review_id for r in rows])).values(notified=1))
        return {"dikirim": len(rows), "penerima": len(recipients), "lewat": provider}

    def _log_notif(self, event, ids):
        if not ids:
            return
        with self.engine.begin() as c:
            c.execute(insert(notif_log), [{"id": uuid.uuid4().hex, "event": event,
                                           "review_id": i, "ts": _now()} for i in ids])

    # ---- 4b. untuk agen notifikasi Copilot Studio (dipanggil lewat endpoint bertoken)
    def negatif_baru(self, batas=20):
        """Ulasan negatif yang belum diemailkan, dengan teks lengkap (maks 1500 karakter), supaya agen Copilot
        Studio menyusun emailnya sendiri. Tanpa nama pengguna."""
        rows = self._pending_negative(_clamp(batas, 1, 50, 20))
        self._log_notif("served", [r.review_id for r in rows])
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(reviews)
                              .where(self._eff() == "negatif", reviews.c.notified == 0)).scalar()
        return {"total_menunggu": total, "ditampilkan": len(rows),
                "ulasan": [{"review_id": r.review_id, "rs": r.rs_name, "rating": r.rating,
                            "tanggal": _tgl(r.published_at),
                            "teks": " ".join(str(r.text or "").split())[:1500]} for r in rows]}

    def tandai_terkirim(self, review_ids):
        """Tandai ulasan sudah diemailkan (dipanggil agen Copilot Studio SETELAH email terkirim)."""
        ids = [str(i) for i in (review_ids or []) if i][:200]
        if not ids:
            return {"ditandai": 0}
        with self.engine.begin() as c:
            n = c.execute(update(reviews).where(reviews.c.review_id.in_(ids)).values(notified=1)).rowcount
        self._log_notif("marked", ids)
        return {"ditandai": n}

    # ---- 4c. ulasan menurut jam tulis (untuk pertanyaan "berapa komentar jam 08.00?")
    def ulasan_per_jam(self, jam, rs="", batas=8):
        """Ulasan yang ditulis pada jam tertentu (WIB). Jam 8 = 08.00-08.59. Bila tidak ada, dipakai jam terdekat
        yang punya ulasan. Mengembalikan jumlah, sebaran sentimen, RS terbanyak, contoh isi, dan sebaran 24 jam."""
        try:
            jam = int(jam)
        except (TypeError, ValueError):
            return {"error": "jam harus angka 0-23"}
        if not 0 <= jam <= 23:
            return {"error": "jam harus 0-23"}
        batas = _clamp(batas, 1, 15, 8)
        q = select(reviews.c.rs_name, reviews.c.rating, reviews.c.text, reviews.c.published_at,
                   reviews.c.label, reviews.c.corrected_label).where(
            reviews.c.published_at.is_not(None), reviews.c.label.is_not(None))
        rs = (rs or "").strip()
        if rs:
            q = q.where(func.lower(reviews.c.rs_name).contains(rs.lower(), autoescape=True))
        with self.engine.connect() as c:
            rows = c.execute(q).all()
        per_jam = {h: [] for h in range(24)}
        for r in rows:
            d = _wib(r.published_at)
            if d:
                per_jam[d.hour].append((d, r))
        sebaran = {f"{h:02d}": len(v) for h, v in per_jam.items()}
        ada = [h for h, v in per_jam.items() if v]
        if not ada:
            return {"zona_waktu": "WIB", "jam_diminta": jam, "filter_rs": rs or None, "jumlah": 0,
                    "catatan": "Tidak ada ulasan berstempel waktu untuk filter ini."}
        jarak = lambda h: min((h - jam) % 24, (jam - h) % 24)  # noqa: E731
        pakai = jam if per_jam[jam] else min(ada, key=lambda h: (jarak(h), -len(per_jam[h]), h))
        items = per_jam[pakai]
        n = len(items)
        lab = Counter(str(r.corrected_label or r.label).capitalize() for _, r in items)
        contoh = sorted(items, key=lambda x: x[0], reverse=True)[:batas]
        return {"zona_waktu": "WIB", "jam_diminta": jam, "jam_dipakai": pakai, "tepat": pakai == jam,
                "selisih_jam": jarak(pakai), "filter_rs": rs or None, "jumlah": n,
                "per_label": dict(lab), "persen": {k: round(100 * v / n, 1) for k, v in lab.items()},
                "rs_terbanyak": [{"rs": a, "jumlah": b} for a, b in Counter(r.rs_name for _, r in items).most_common(5)],
                "contoh": [{"rs": r.rs_name, "rating": r.rating, "tanggal": _tgl(r.published_at),
                            "jam_wib": f"{d.hour:02d}.{d.minute:02d}", "label": str(r.corrected_label or r.label).capitalize(),
                            "teks": " ".join(str(r.text or "").split())[:300]} for d, r in contoh],
                "sebaran_jam_wib": sebaran,
                "catatan": "Jam berasal dari stempel waktu publikasi di data (UTC, dikonversi ke WIB). Google Maps hanya "
                           "menampilkan waktu relatif, jadi jam bisa berupa perkiraan. Label = prediksi SVM "
                           "(atau koreksi manusia bila ada)."}

    # ---- 4d. ringkasan per RS: rating + sebaran sentimen, dan (bila ada topik) berapa ulasan yang menyebut topik itu
    def ringkasan_rs(self, topik="", rs="", batas=8):
        """Per RS: jumlah ulasan, rating rata-rata, sebaran Positif/Netral/Negatif. Bila topik diisi (mis. "IGD",
        "antrean farmasi"), ditambah jumlah ulasan yang menyebut topik itu beserta sebaran sentimennya."""
        batas = _clamp(batas, 1, 25, 8)
        q = select(reviews.c.rs_name, reviews.c.rating, reviews.c.text, reviews.c.label,
                   reviews.c.corrected_label).where(reviews.c.label.is_not(None))
        rs = (rs or "").strip()
        if rs:
            q = q.where(func.lower(reviews.c.rs_name).contains(rs.lower(), autoescape=True))
        with self.engine.connect() as c:
            rows = c.execute(q).all()
        kunci = _kata_kunci(topik) if (topik or "").strip() else []
        pola = _pola_topik(kunci)
        agg = {}
        for r in rows:
            a = agg.setdefault(r.rs_name, {"n": 0, "rs": 0.0, "rn": 0, "lab": Counter(), "tn": 0, "tlab": Counter()})
            lab = str(r.corrected_label or r.label).capitalize()
            a["n"] += 1
            a["lab"][lab] += 1
            if isinstance(r.rating, (int, float)) and math.isfinite(r.rating):
                a["rs"] += r.rating
                a["rn"] += 1
            if pola and pola.search(r.text or ""):
                a["tn"] += 1
                a["tlab"][lab] += 1
        pct = lambda x, n: round(100 * x / n, 1) if n else None  # noqa: E731
        hasil = []
        for nama, a in agg.items():
            item = {"rs": nama, "jumlah_ulasan": a["n"],
                    "rating_rata2": round(a["rs"] / a["rn"], 2) if a["rn"] else None,
                    "positif": a["lab"]["Positif"], "netral": a["lab"]["Netral"], "negatif": a["lab"]["Negatif"],
                    "persen_positif": pct(a["lab"]["Positif"], a["n"]), "persen_negatif": pct(a["lab"]["Negatif"], a["n"]),
                    "sampel_kecil": a["n"] < 20}
            if pola:
                item.update({"menyebut_topik": a["tn"], "topik_positif": a["tlab"]["Positif"],
                             "topik_netral": a["tlab"]["Netral"], "topik_negatif": a["tlab"]["Negatif"],
                             "persen_positif_topik": pct(a["tlab"]["Positif"], a["tn"])})
            hasil.append(item)
        if pola:
            hasil.sort(key=lambda x: (-x["menyebut_topik"], -(x["persen_positif_topik"] or 0)))
        else:
            hasil.sort(key=lambda x: (x["sampel_kecil"], -(x["rating_rata2"] or 0)))
        return {"topik": topik or None, "kata_kunci": kunci, "jumlah_rs": len(hasil), "hasil": hasil[:batas],
                "catatan": "Label = prediksi SVM (atau koreksi manusia). Ulasan Google Maps adalah sampel, bukan data resmi."}

    # ---- 5. antrean ulasan ragu (dicek manusia lewat GET /agent/ragu) dan koreksi manusia
    def lihat_ulasan_ragu(self, batas=10):
        with self.engine.connect() as c:
            rows = c.execute(select(reviews.c.review_id, reviews.c.rs_name, reviews.c.rating, reviews.c.text,
                                    reviews.c.label, reviews.c.confidence)
                             .where(reviews.c.corrected_label.is_(None), reviews.c.label.is_not(None))
                             .order_by(reviews.c.confidence).limit(_clamp(batas, 1, 20, 10))).all()
        return {"jumlah": len(rows),
                "ulasan": [{"review_id": r.review_id, "rs": r.rs_name, "rating": r.rating,
                            "teks": (r.text or "")[:180], "label_svm": r.label, "confidence": r.confidence}
                           for r in rows]}

    def simpan_koreksi_manusia(self, review_id, label):
        """Dipakai endpoint POST /koreksi. Label manusia selalu menang."""
        lab = self._canon(label)
        if not lab:
            return {"error": "label bukan kelas valid"}
        with self.engine.begin() as c:
            n = c.execute(update(reviews).where(reviews.c.review_id == review_id)
                          .values(corrected_label=lab, label_source="manusia")).rowcount
        return {"tersimpan": n}

    # ---- 6. retrain dengan gerbang evaluasi
    def _read_labeled(self, path):
        """Baca CSV berlabel -> DataFrame[TEXT_COL, LABEL_COL] berisi teks BERSIH.
        Urutan: file *_clean.csv (hasil prepare_training_data.py) > kolom clean_text > kolom text mentah
        yang dibersihkan dengan preprocess() yang sama dengan yang dipakai /predict."""
        import pandas as pd
        clean_path = path[:-4] + "_clean.csv" if path.endswith(".csv") else path
        src = clean_path if os.path.exists(clean_path) else path
        df = pd.read_csv(src)
        if TEXT_COL in df.columns:
            txt = df[TEXT_COL].astype(str)
        elif RAW_TEXT_COL in df.columns:
            txt = df[RAW_TEXT_COL].astype(str).map(self.preprocess)
        else:
            raise ValueError(f"{src}: butuh kolom '{TEXT_COL}' atau '{RAW_TEXT_COL}', ada {list(df.columns)}")
        out = pd.DataFrame({TEXT_COL: txt, LABEL_COL: df[LABEL_COL]}).dropna()
        out = out[out[TEXT_COL].str.strip().ne("") & out[TEXT_COL].ne("nan")]
        return out.reset_index(drop=True)
        
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    def retrain_model(self):
        import pandas as pd
        from sklearn.base import clone
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import LinearSVC

        t0 = time.time()
        with self.engine.connect() as c:
            corr = c.execute(select(reviews.c.clean, reviews.c.corrected_label)
                             .where(reviews.c.corrected_label.is_not(None), reviews.c.clean.is_not(None))).all()
        last = int(self._kv_get("last_retrain_count", 0))
        if len(corr) - last < MIN_NEW_CORRECTIONS:
            return {"dilewati": f"koreksi baru {len(corr) - last} < syarat {MIN_NEW_CORRECTIONS}"}
        if not (os.path.exists(TRAIN_POOL) and os.path.exists(TEST_SET)):
            return {"error": f"butuh {TRAIN_POOL} dan {TEST_SET} (kolom {TEXT_COL}, {LABEL_COL}); test set harus label manusia"}

        cur_pipe, cur_le, cur_metrics = self.get_model()
        canon = {str(c).lower(): str(c) for c in cur_le.classes_}
        norm = lambda s: canon.get(str(s).strip().lower(), str(s).strip())  # noqa: E731

        base = self._read_labeled(TRAIN_POOL)
        test = self._read_labeled(TEST_SET)
        extra = pd.DataFrame({TEXT_COL: [r.clean for r in corr], LABEL_COL: [r.corrected_label for r in corr]})
        train = pd.concat([extra, base], ignore_index=True)
        train[LABEL_COL] = train[LABEL_COL].map(norm)
        test[LABEL_COL] = test[LABEL_COL].map(norm)
        train = train[~train[TEXT_COL].isin(set(test[TEXT_COL]))].drop_duplicates(TEXT_COL)   # cegah kebocoran test set

        le = LabelEncoder().fit(train[LABEL_COL])
        y = le.transform(train[LABEL_COL])
        min_cnt = min(Counter(y).values())
        if min_cnt < 2:
            return {"error": "salah satu kelas hanya punya 1 contoh; retrain dibatalkan"}
        cv = max(2, min(5, min_cnt))
        cand = clone(cur_pipe).fit(train[TEXT_COL], y)      
        cv_f1 = float(cross_val_score(clone(cur_pipe), train[TEXT_COL], y,
                                      cv=cv, scoring="f1_macro", n_jobs=1).mean())

        texts, truth = list(test[TEXT_COL]), list(test[LABEL_COL])
        cand_pred = list(le.inverse_transform(cand.predict(texts)))
        cur_pred = list(cur_le.inverse_transform(cur_pipe.predict(texts)))
        f1c = f1_score(truth, cand_pred, average="macro", zero_division=0)
        f1o = f1_score(truth, cur_pred, average="macro", zero_division=0)
        neg = canon.get("negatif", "Negatif")
        nr = lambda p: recall_score([t == neg for t in truth], [x == neg for x in p], zero_division=0)  # noqa: E731
        rc, ro = nr(cand_pred), nr(cur_pred)

        ok = f1c > f1o + MIN_F1_GAIN and rc >= ro - MAX_NEG_RECALL_DROP
        result = {"f1_model_lama": round(f1o, 4), "f1_kandidat": round(float(f1c), 4),
                  "recall_negatif_lama": round(ro, 4), "recall_negatif_kandidat": round(rc, 4),
                  "contoh_latih": int(len(train)), "contoh_uji": len(truth), "dipasang": bool(ok)}
        if ok:
            metrics = {"accuracy": round(accuracy_score(truth, cand_pred), 4),
                       "precision_macro": round(precision_score(truth, cand_pred, average="macro", zero_division=0), 4),
                       "recall_macro": round(recall_score(truth, cand_pred, average="macro", zero_division=0), 4),
                       "f1_macro": round(float(f1c), 4), "cv_f1_macro": round(cv_f1, 4)}
            result["versi"] = save_model(self.engine, cand, le, metrics)
            self.deploy(cand, le, metrics)
            result["relabel"] = self._relabel_all()
            result["durasi_detik"] = round(time.time() - t0, 1)
        self._kv_set("last_retrain_count", len(corr))          # sukses atau tidak, jangan diulang tiap run
        self._kv_set("last_retrain_result", json.dumps(result))
        return result

    def _relabel_all(self):
        """Label ulang ulasan dengan model baru (corrected_label tidak tersentuh). Maks 20.000 baris per retrain."""
        return self.label_ulasan(batas=20000, semua=True).get("dilabeli", 0)

    # ---- dispatch langkah rutinitas (nama langkah -> method)
    def call(self, name, args=None):
        args = args if isinstance(args, dict) else {}
        fn = {"status_data": self.status_data, "ambil_ulasan_baru": self.ambil_ulasan_baru,
              "label_ulasan": self.label_ulasan,
              "lihat_ulasan_negatif_belum_dikirim": self.lihat_ulasan_negatif_belum_dikirim,
              "kirim_email_negatif": self.kirim_email_negatif,
              "retrain_model": self.retrain_model}.get(name)
        if not fn:
            return {"error": f"langkah '{name}' tidak ada"}
        if name == "label_ulasan":
            args = {k: v for k, v in args.items() if k == "batas"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"argumen tidak valid: {e}"}
        except Exception as e:
            return {"error": f"langkah gagal: {type(e).__name__}: {e}"}

    # ---- 7. pemantauan kesehatan (dipanggil setelah tiap run oleh run_routine)
    def _alert_ok(self, key):
        last = self._kv_get("alert_" + key)
        if last:
            try:
                if datetime.now(timezone.utc) - datetime.fromisoformat(last.replace("Z", "+00:00")) < timedelta(hours=ALERT_COOLDOWN_H):
                    return False
            except ValueError:
                pass
        self._kv_set("alert_" + key, _now())
        return True

    def _kirim_peringatan(self, jenis, isi):
        to = (os.environ.get("ALERT_TO") or os.environ.get("NOTIFY_TO") or "").strip()
        prov = _email_provider()
        if not to or prov == "smtp":
            print("[health] peringatan tidak dikirim:", isi[:200], flush=True)
            return
        try:
            self.http_send(prov, [x.strip() for x in to.split(",") if x.strip()], f"[Geosentimen] PERINGATAN sistem: {jenis}", isi)
        except Exception as e:
            print("[health] gagal kirim peringatan:", e, flush=True)

    def _apify_usage(self):
        token = os.environ.get("APIFY_TOKEN")
        if not token:
            return None
        import httpx
        r = httpx.get("https://api.apify.com/v2/users/me/limits", headers={"Authorization": f"Bearer {token}"}, timeout=15)
        if r.status_code != 200:
            return None
        d = r.json().get("data", {})
        used, cap = d.get("current", {}).get("monthlyUsageUsd"), d.get("limits", {}).get("maxMonthlyUsageUsd")
        if used is None or not cap:
            print("[health] field kredit Apify tidak ditemukan:", list(d.keys()), flush=True)
            return None
        return {"used": float(used), "cap": float(cap), "pct": round(100 * float(used) / float(cap), 1)}

    def pantau_kesehatan(self):
        """Dipanggil setelah tiap run. Mengirim email peringatan (satu per jenis per 12 jam)."""
        masalah = []
        with self.engine.connect() as c:
            runs = c.execute(select(agent_runs.c.status, agent_runs.c.report).order_by(agent_runs.c.started_at.desc()).limit(2)).all()
        if len(runs) == 2 and all(r.status != "ok" for r in runs):
            masalah.append(("run_gagal", "Run gagal 2 kali berturut-turut. Terakhir: " + (runs[0].report or "")[:300]))
        try:
            u = self._apify_usage()
            if u and u["pct"] >= 80:
                masalah.append(("kredit_apify", f"Kredit Apify terpakai {u['pct']}% (${u['used']:.2f} dari ${u['cap']:.2f})"))
        except Exception as e:
            print("[health] cek Apify gagal:", e, flush=True)
        kirim = [m for m in masalah if self._alert_ok(m[0])]
        if kirim:
            self._kirim_peringatan(", ".join(k for k, _ in kirim), "\n\n".join(t for _, t in kirim))
        return [t for _, t in masalah]

    def status_kesehatan(self, maks_jam=13):
        with self.engine.connect() as c:
            ok = c.execute(select(agent_runs.c.finished_at).where(agent_runs.c.status == "ok")
                           .order_by(agent_runs.c.finished_at.desc()).limit(1)).scalar()
        umur = None
        if ok:
            umur = (datetime.now(timezone.utc) - datetime.fromisoformat(ok.replace("Z", "+00:00"))).total_seconds() / 3600
        return {"ok": umur is not None and umur <= maks_jam, "jam_sejak_run_sukses": None if umur is None else round(umur, 1)}


# ------------------------------------------------------------------ rutinitas (tanpa LLM)
_LOCK = asyncio.Lock()


def ops_running():
    return _LOCK.locked()


def _jalankan(tools, log):
    """Langkah tetap. Ulasan baru otomatis dilabeli SVM di dalam ambil_ulasan_baru."""
    langkah = ["ambil_ulasan_baru"]
    lewat_copilot = _notify_via_copilot()
    if not lewat_copilot:
        langkah.append("kirim_email_negatif")
    if not RETRAIN_OLEH_COPILOT:
        langkah.append("retrain_model")
    gagal = []
    for nama in langkah:
        hasil = tools.call(nama, {})
        print(f"[ops] {nama} -> {str(hasil)[:150]}", flush=True)
        log.append({"alat": nama, "hasil": str(hasil)[:300]})
        if isinstance(hasil, dict) and "error" in hasil:
            gagal.append(nama)
    if lewat_copilot:
        log.append({"alat": "kirim_email_negatif",
                    "hasil": "dilewati: email dikirim agen Copilot Studio (NOTIFY_MODE=copilot)"})
    if RETRAIN_OLEH_COPILOT:
        log.append({"alat": "retrain_model", "hasil": "dilewati: dijalankan agen Copilot Studio (RETRAIN_MODE=copilot)"})
    ringkasan = "Rutinitas selesai:\n" + "\n".join(f"- {x['alat']}: {x['hasil'][:200]}" for x in log)
    if gagal:
        raise RuntimeError(f"Langkah gagal: {', '.join(gagal)}\n{ringkasan}")
    return ringkasan

async def run_routine(tools, *_abaikan, **_kw):
    """Jalankan satu rutinitas dan catat hasilnya di tabel agent_runs (argumen tambahan diabaikan agar
    pemanggil lama tetap kompatibel)."""
    if _LOCK.locked():
        return
    async with _LOCK:
        run_id = uuid.uuid4().hex[:12]
        with tools.engine.begin() as c:
            # Karena kunci proses ini bebas, baris 'running' yang masih ada pasti sisa proses lama yang mati
            # (restart/deploy Render, kehabisan memori, atau service tidur di tengah run).
            c.execute(update(agent_runs).where(agent_runs.c.status == "running").values(
                status="terputus", finished_at=_now(),
                report="Run terputus: proses server berhenti sebelum selesai (restart/deploy, kehabisan memori, atau service tidur)."))
            c.execute(insert(agent_runs).values(run_id=run_id, started_at=_now(), status="running"))
        status, report, log = "ok", "", []
        try:
            report = await asyncio.to_thread(_jalankan, tools, log)
        except Exception as e:
            status, report = "error", f"{type(e).__name__}: {e}"
        finally:
            with tools.engine.begin() as c:
                c.execute(update(agent_runs).where(agent_runs.c.run_id == run_id)
                          .values(finished_at=_now(), status=status, report=report[:4000],
                                  log=json.dumps(log, ensure_ascii=False)[:20000]))
        try:
            await asyncio.to_thread(tools.pantau_kesehatan)
        except Exception as e:
            print("[health]", e, flush=True)
        return run_id