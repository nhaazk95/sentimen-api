"""ops_agent.py — agen operasional Geosentimen RS Banyumas (versi Agentic Full)."""
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
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.message import EmailMessage

import joblib
import numpy as np
from sqlalchemy import (Column, Float, Integer, LargeBinary, MetaData, String, Table,
                        Text, bindparam, create_engine, func, insert, select, update)

from agent import HospitalTools, TOPIK, topik_dari_teks


class AgentError(Exception):
    def __init__(self, status, detail=""):
        super().__init__(f"{status}: {str(detail)[:200]}")
        self.status = status


# ------------------------------------------------------------------ konfigurasi
PLACES_FILE = os.environ.get("PLACES_FILE", "places.json")
APIFY_ACTOR = os.environ.get("APIFY_ACTOR", "compass/crawler-google-places")
APIFY_MAX_CHARGE_USD = os.environ.get("APIFY_MAX_CHARGE_USD", "2")
DEFAULT_MAX_REVIEWS = 5
NOTIFY_MAX_AGE_DAYS = 14
INDONESIA_BBOX = {"lat_min": -11, "lat_max": 6, "lng_min": 95, "lng_max": 141}
TRAIN_POOL = os.environ.get("TRAIN_POOL", "data/training/train_pool.csv")
TEST_SET = os.environ.get("TEST_SET", "data/training/test_set.csv")
TEXT_COL = os.environ.get("TEXT_COL", "clean_text")
RAW_TEXT_COL = os.environ.get("RAW_TEXT_COL", "text")
LABEL_COL = os.environ.get("LABEL_COL", "label")
MIN_NEW_CORRECTIONS = int(os.environ.get("MIN_NEW_CORRECTIONS", "30"))
MIN_F1_GAIN = float(os.environ.get("MIN_F1_GAIN", "0.0"))
MAX_NEG_RECALL_DROP = 0.05
MAX_STEPS = 8
TOOL_RESULT_CHARS = 3500
GROQ_SEEN = {}
ALERT_COOLDOWN_H = 12
_ANON = re.compile(r"(\+?\d[\d\s\-]{6,}\d|[\w.+-]+@[\w-]+\.[\w.-]+)")

# ------------------------------------------------------------------ database
md = MetaData()
reviews = Table(
    "reviews", md,
    Column("review_id", String(255), primary_key=True),
    Column("rs_name", String(255)), Column("place_id", String(255)),
    Column("rating", Float), Column("text", Text), Column("clean", Text),
    Column("label", String(20)), Column("confidence", Float),
    Column("corrected_label", String(20)), Column("label_source", String(10)),
    Column("published_at", String(40)), Column("scraped_at", String(40)),
    Column("notified", Integer, nullable=False, default=0),
    Column("username", String(255)), Column("address", Text), Column("lat", Float), Column("lon", Float),
    Column("processed_at", String(40)),
    Column("legacy", Integer, nullable=False, default=0),
    Column("topics", Text),
)
models = Table("models", md,
               Column("version", String(40), primary_key=True),
               Column("created_at", String(40)),
               Column("blob", LargeBinary), Column("metrics", Text),
               Column("active", Integer, default=0))
agent_runs = Table("agent_runs", md,
                   Column("run_id", String(40), primary_key=True),
                   Column("started_at", String(40)),
                   Column("finished_at", String(40)),
                   Column("status", String(20)), Column("report", Text),
                   Column("log", Text))
kv = Table("kv", md, Column("k", String(80), primary_key=True), Column("v", Text))
rs_info = Table("rs_info", md, Column("rs_name", String(255), primary_key=True),
                Column("phone", String(60)), Column("website", String(300)),
                Column("categories", Text), Column("hours", Text),
                Column("extra", Text), Column("updated_at", String(40)))
chat_log = Table("chat_log", md, Column("id", Integer, primary_key=True, autoincrement=True),
                 Column("ts", String(40)), Column("question", Text), Column("tools", Text),
                 Column("empty_result", Integer), Column("used_gps", Integer),
                 Column("mode", String(10)), Column("ms", Integer))
monthly_reports = Table(
    "monthly_reports", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("rs_name", String(255)),
    Column("period", String(7)),
    Column("total", Integer),
    Column("positif", Integer), Column("netral", Integer), Column("negatif", Integer),
    Column("avg_rating", Float),
    Column("top_complaints", Text), Column("top_praises", Text),
    Column("sample_negative", Text),
    Column("prev_period", String(7)), Column("delta_negatif_pct", Float),
    Column("generated_at", String(40)),
    Column("sent", Integer, nullable=False, default=0),
)


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
    return None if isinstance(v, float) and not math.isfinite(v) else v


# ------------------------------------------------------------------ model
def save_model(engine, pipeline, label_encoder, metrics):
    buf = io.BytesIO()
    joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder}, buf)
    version = "v" + time.strftime("%Y%m%d%H%M%S")
    with engine.begin() as c:
        c.execute(update(models).values(active=0))
        c.execute(insert(models).values(version=version, created_at=_now(),
                                        blob=buf.getvalue(),
                                        metrics=json.dumps(metrics), active=1))
    return version


def load_active_model(engine):
    with engine.connect() as c:
        row = c.execute(select(models.c.blob, models.c.metrics)
                        .where(models.c.active == 1)
                        .order_by(models.c.created_at.desc())).first()
    if not row:
        return None
    obj = joblib.load(io.BytesIO(row.blob))
    return obj["pipeline"], obj["label_encoder"], json.loads(row.metrics)


def last_run_info(engine):
    with engine.connect() as c:
        r = c.execute(select(agent_runs.c.status, agent_runs.c.started_at)
                      .order_by(agent_runs.c.started_at.desc()).limit(1)).first()
        ok = c.execute(select(agent_runs.c.finished_at)
                       .where(agent_runs.c.status == "ok")
                       .order_by(agent_runs.c.finished_at.desc()).limit(1)).scalar()
    if not r:
        return {"status": "unknown", "conclusion": None, "created_at": None, "last_run_utc": ok}
    return {"status": "in_progress" if r.status == "running" else "completed",
            "conclusion": None if r.status == "running" else ("success" if r.status == "ok" else "failure"),
            "created_at": r.started_at, "last_run_utc": ok}


def log_chat(engine, question, tools, empty, used_gps, mode, ms):
    q = _ANON.sub("[disamarkan]", (question or "")[:500])
    with engine.begin() as c:
        c.execute(insert(chat_log).values(ts=_now(), question=q,
                                          tools=json.dumps(tools, ensure_ascii=False)[:500],
                                          empty_result=int(empty), used_gps=int(used_gps),
                                          mode=mode, ms=ms))


def list_chat_log(engine, batas=50, kosong=False):
    q = select(chat_log).order_by(chat_log.c.id.desc()).limit(_clamp(batas, 1, 200, 50))
    if kosong:
        q = q.where(chat_log.c.empty_result == 1)
    with engine.connect() as c:
        return [dict(r) for r in c.execute(q).mappings().all()]


def export_reviews(engine):
    with engine.connect() as c:
        rows = c.execute(select(reviews).where(reviews.c.label.is_not(None))
                         .order_by(reviews.c.published_at.desc())).mappings().all()
    return [{"Nama RS": r["rs_name"], "Username": r["username"], "Rating": _num(r["rating"]),
             "Waktu Ulasan": r["published_at"], "Lokasi Tempat": r["address"],
             "Latitude": _num(r["lat"]), "Longitude": _num(r["lon"]),
             "Isi Ulasan": r["text"], "id": r["review_id"],
             "Sentimen_Prediksi": r["corrected_label"] or r["label"],
             "Confidence": _num(r["confidence"]), "Processed_At": r["processed_at"],
             "Teks_Bersih": r["clean"],
             "Label_Manual": r["corrected_label"] if r["label_source"] == "manusia" else None,
             "Notified": bool(r["notified"]),
             "Topik": json.loads(r["topics"] or "[]") if r.get("topics") else []}
            for r in rows]


# ------------------------------------------------------------------ util apify/email
def _get_field(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in key).lstrip("_")
    return getattr(obj, snake, None) if hasattr(obj, snake) else getattr(obj, key, None)


def _apify_fetch(token, run_input):
    from apify_client import ApifyClient
    client = ApifyClient(token)
    run = client.actor(APIFY_ACTOR).call(run_input=run_input,
                                         max_total_charge_usd=Decimal(APIFY_MAX_CHARGE_USD))
    return list(client.dataset(_get_field(run, "defaultDatasetId")).iterate_items())


def _smtp_send(user, password, recipients, msg):
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(user, password)
        s.send_message(msg, from_addr=user, to_addrs=recipients)


def _http_send(provider, recipients, subject, text, html=None):
    import httpx
    if provider == "resend":
        r = httpx.post("https://api.resend.com/emails", timeout=30,
                       headers={"Authorization": f"Bearer {os.environ.get('RESEND_API_KEY')}"},
                       json={"from": os.environ.get("EMAIL_FROM") or "Geosentimen <onboarding@resend.dev>",
                             "to": recipients, "subject": subject, "text": text,
                             **({"html": html} if html else {})})
    elif provider == "brevo":
        r = httpx.post("https://api.brevo.com/v3/smtp/email", timeout=30,
                       headers={"api-key": os.environ.get("BREVO_API_KEY", ""),
                                "accept": "application/json"},
                       json={"sender": {"email": os.environ.get("EMAIL_FROM"),
                                        "name": "Geosentimen RS Banyumas"},
                             "to": [{"email": x} for x in recipients],
                             "subject": subject, "textContent": text,
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


BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]


def _wib(iso=None):
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


# ------------------------------------------------------------------ email negatif harian
def _susun_email(rows, ringkasan="", urgensi="RENDAH", rekomendasi=""):
    esc = _html.escape
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: str(r.published_at or ""), reverse=True):
        by.setdefault(r.rs_name or "RS tidak diketahui", []).append(r)
    n, k = len(rows), len(by)
    now = _wib()
    waktu = f"{now.day} {BULAN[now.month - 1]} {now.year}, {now.hour:02d}.{now.minute:02d} WIB"
    ringkasan = " ".join((ringkasan or "").split())[:800]
    dashboard = os.environ.get("DASHBOARD_URL", "")
    
    warna_header = "#dc2626" if urgensi == "TINGGI" else ("#f59e0b" if urgensi == "SEDANG" else "#1e3a8a")
    label_urgensi = f"🚨 PRIORITAS {urgensi} 🚨" if urgensi == "TINGGI" else f"PRIORITAS {urgensi}"
    
    subject = f"[{urgensi}] {n} Ulasan Negatif Baru - {k} RS"

    # --- SUSUNAN TEKS BIASA ---
    t = [f"ULASAN NEGATIF BARU ({urgensi})", waktu,
         f"{n} ulasan negatif di {k} rumah sakit", ""]
    if ringkasan:
        t += ["RINGKASAN KELUHAN", ringkasan, ""]
    if rekomendasi:
        t += ["REKOMENDASI TINDAKAN", rekomendasi, ""]
        
    for rs, items in by.items():
        t.append(f"== {rs} ({len(items)} ulasan) ==")
        for r in items:
            t += [f"[{_bintang(r.rating)[1]}] {_tgl(r.published_at)}", _potong(r.text), ""]
            
    t.append("Catatan: Sistem ini menggunakan Agentic AI. Prediksi sentimen dibuat otomatis oleh model SVM.")
    if dashboard:
        t.append(f"Dashboard: {dashboard}")
        
    teks = "\n".join(t)

    # --- SUSUNAN HTML ---
    def kartu(r):
        bintang, skor = _bintang(r.rating)
        isi = esc(_potong(r.text)).replace("\n", "<br>")
        return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
                'style="margin:10px 0;background:#fff7f7;border:1px solid #fecaca;'
                'border-left:4px solid #dc2626;border-radius:6px;">'
                '<tr><td style="padding:12px 14px;font-family:Arial,Helvetica,sans-serif;">'
                f'<div style="font-size:13px;font-weight:700;color:#b91c1c;letter-spacing:1px;">'
                f'{bintang}<span style="font-weight:400;letter-spacing:0;color:#6b7280;">'
                f'&nbsp; {skor} &bull; {esc(_tgl(r.published_at))}</span></div>'
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
               
    box_rekomendasi = ""
    if rekomendasi:
        box_rekomendasi = ('<tr><td style="padding:10px 28px 0 28px;font-family:Arial,Helvetica,sans-serif;">'
                           '<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:12px 14px;">'
                           '<div style="font-size:12px;font-weight:700;color:#1e40af;letter-spacing:.5px;">REKOMENDASI TINDAKAN</div>'
                           f'<div style="font-size:14px;line-height:1.6;color:#1e3a8a;margin-top:4px;">{esc(rekomendasi)}</div>'
                           '</div></td></tr>')

    link = (f'<div style="margin-top:14px;"><a href="{esc(dashboard)}" '
            'style="display:inline-block;background:#1e3a8a;color:#ffffff;text-decoration:none;'
            'padding:10px 18px;border-radius:6px;font-weight:700;">Buka dashboard</a></div>'
            if dashboard else "")
            
    html = (
        '<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
        '<body style="margin:0;padding:0;background:#f3f4f6;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;">'
        '<tr><td align="center" style="padding:24px 12px;">'
        '<table role="presentation" width="600" cellpadding="0" cellspacing="0" '
        'style="width:100%;max-width:600px;background:#ffffff;border-radius:10px;overflow:hidden;">'
        f'<tr><td style="background:{warna_header};padding:22px 28px;font-family:Arial,Helvetica,sans-serif;">'
        '<div style="font-size:12px;color:#ffffff;letter-spacing:1px;opacity:0.8;">DASHBOARD SENTIMEN RS BANYUMAS</div>'
        f'<div style="font-size:22px;font-weight:700;color:#ffffff;margin-top:4px;">{label_urgensi}</div>'
        f'<div style="font-size:13px;color:#ffffff;opacity:0.9;margin-top:6px;">{esc(waktu)}</div></td></tr>'
        '<tr><td style="padding:18px 28px 0 28px;font-family:Arial,Helvetica,sans-serif;">'
        f'<span style="font-size:30px;font-weight:700;color:#dc2626;">{n}</span>'
        f'<span style="font-size:14px;color:#374151;"> ulasan negatif di {k} rumah sakit</span></td></tr>'
        + box + box_rekomendasi + "".join(bagian) +
        '<tr><td style="padding:22px 28px 24px 28px;font-family:Arial,Helvetica,sans-serif;">'
        '<div style="border-top:1px solid #e5e7eb;padding-top:14px;font-size:12px;line-height:1.6;color:#6b7280;">'
        'Sistem ini menggunakan Agentic AI. Prediksi sentimen dibuat oleh model SVM dan dapat keliru. Mohon periksa ulasan.'
        + link + '</div></td></tr>'
        '</table></td></tr></table></body></html>')
    return subject, teks, html


# ------------------------------------------------------------------ email laporan bulanan
def _susun_email_bulanan(period, rows, prev_rows=None):
    esc = _html.escape
    prev_rows = prev_rows or {}
    now = _wib()
    waktu = f"{now.day} {BULAN[now.month - 1]} {now.year}, {now.hour:02d}.{now.minute:02d} WIB"
    y, m = period.split("-")
    nama_periode = f"{BULAN[int(m) - 1]} {y}"
    dashboard = os.environ.get("DASHBOARD_URL", "")
    total = sum(r["total"] for r in rows)
    pos = sum(r["positif"] for r in rows)
    net = sum(r["netral"] for r in rows)
    neg = sum(r["negatif"] for r in rows)
    pct = lambda x: f"{(100.0 * x / total):.1f}%" if total else "0%"  # noqa: E731
    subject = f"[Geosentimen] Laporan Bulanan {nama_periode} — {len(rows)} RS"

    teks = [f"LAPORAN BULANAN — {nama_periode.upper()}", waktu,
            f"Total {total} ulasan · {pct(pos)} positif · {pct(net)} netral · {pct(neg)} negatif", ""]
    for r in rows:
        teks.append(f"== {r['rs_name']} ==")
        teks.append(f"{r['total']} ulasan · {r['positif']} positif · {r['netral']} netral · "
                    f"{r['negatif']} negatif · rating {r['avg_rating'] or '-'}")
        if r["top_complaints"]:
            teks.append("Keluhan: " + ", ".join(f"{c['topik']} ({c['count']}x)"
                                                for c in r["top_complaints"][:5]))
        if r["top_praises"]:
            teks.append("Pujian: " + ", ".join(f"{c['topik']} ({c['count']}x)"
                                               for c in r["top_praises"][:5]))
        prev = prev_rows.get(r["rs_name"])
        if prev:
            delta = r["negatif_pct"] - prev["negatif_pct"]
            teks.append(f"Delta negatif vs {prev['period']}: {delta:+.1f}%")
        teks.append("")

    def blok_rs(r):
        prev = prev_rows.get(r["rs_name"])
        delta_html = ""
        if prev:
            delta = r["negatif_pct"] - prev["negatif_pct"]
            warna = "#16a34a" if delta <= 0 else "#dc2626"
            arah = "▼" if delta <= 0 else "▲"
            delta_html = (f'<span style="color:{warna};font-weight:700;">{arah} {abs(delta):.1f}%</span> '
                          f'<span style="color:#6b7280;font-size:12px;">vs {esc(prev["period"])}</span>')
        bar = ""
        if r["total"]:
            p_pos = 100 * r["positif"] / r["total"]
            p_net = 100 * r["netral"] / r["total"]
            p_neg = 100 * r["negatif"] / r["total"]
            bar = ('<div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin:10px 0 8px 0;">'
                   f'<div style="width:{p_pos:.1f}%;background:#16a34a;"></div>'
                   f'<div style="width:{p_net:.1f}%;background:#f59e0b;"></div>'
                   f'<div style="width:{p_neg:.1f}%;background:#dc2626;"></div></div>')
        keluhan = ", ".join(f"{esc(c['topik'])} ({c['count']}x)" for c in (r["top_complaints"] or [])[:4]) or "-"
        pujian = ", ".join(f"{esc(c['topik'])} ({c['count']}x)" for c in (r["top_praises"] or [])[:4]) or "-"
        kutipan = ""
        if r["sample_negative"]:
            kutipan = ('<div style="margin-top:8px;font-style:italic;color:#6b7280;font-size:13px;">"'
                       + esc(_potong(r["sample_negative"][0], 180)) + '"</div>')
        return (
            '<tr><td style="padding:18px 28px 0 28px;font-family:Arial,Helvetica,sans-serif;">'
            '<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px;">'
            f'<div style="font-size:16px;font-weight:700;color:#111827;">{esc(r["rs_name"])}'
            f'<span style="float:right;color:#6b7280;font-size:13px;font-weight:400;">'
            f'{"★" * int(round(r["avg_rating"] or 0))} {r["avg_rating"] or "-"}</span></div>'
            f'<div style="font-size:12px;color:#6b7280;margin-top:2px;">{r["total"]} ulasan'
            f' &bull; {r["positif"]} positif &bull; {r["netral"]} netral &bull; {r["negatif"]} negatif</div>'
            + bar +
            f'<div style="font-size:13px;color:#374151;margin-top:6px;"><b>Keluhan:</b> {keluhan}</div>'
            f'<div style="font-size:13px;color:#374151;margin-top:2px;"><b>Pujian:</b> {pujian}</div>'
            f'<div style="font-size:13px;margin-top:6px;">{delta_html}</div>'
            + kutipan +
            '</div></td></tr>')

    link = (f'<div style="margin-top:14px;"><a href="{esc(dashboard)}" '
            'style="display:inline-block;background:#1e3a8a;color:#ffffff;text-decoration:none;'
            'padding:10px 18px;border-radius:6px;font-weight:700;">Buka dashboard</a></div>'
            if dashboard else "")
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
        f'<div style="font-size:22px;font-weight:700;color:#ffffff;margin-top:4px;">Laporan {esc(nama_periode)}</div>'
        f'<div style="font-size:13px;color:#dbeafe;margin-top:6px;">Digenerate: {esc(waktu)}</div></td></tr>'
        '<tr><td style="padding:18px 28px 0 28px;font-family:Arial,Helvetica,sans-serif;">'
        f'<div style="font-size:30px;font-weight:700;color:#1e3a8a;">{total}</div>'
        f'<div style="font-size:14px;color:#374151;">ulasan dari {len(rows)} rumah sakit</div>'
        '<div style="display:flex;height:12px;border-radius:6px;overflow:hidden;margin-top:10px;">'
        f'<div style="width:{(100 * pos / total) if total else 0:.1f}%;background:#16a34a;"></div>'
        f'<div style="width:{(100 * net / total) if total else 0:.1f}%;background:#f59e0b;"></div>'
        f'<div style="width:{(100 * neg / total) if total else 0:.1f}%;background:#dc2626;"></div></div>'
        f'<div style="font-size:12px;color:#6b7280;margin-top:6px;">'
        f'{pct(pos)} positif &bull; {pct(net)} netral &bull; {pct(neg)} negatif</div>'
        '</td></tr>'
        + "".join(blok_rs(r) for r in rows) +
        '<tr><td style="padding:22px 28px 24px 28px;font-family:Arial,Helvetica,sans-serif;">'
        '<div style="border-top:1px solid #e5e7eb;padding-top:14px;font-size:12px;line-height:1.6;color:#6b7280;">'
        'Prediksi sentimen dibuat otomatis oleh model SVM dan dapat keliru. Mohon periksa isi ulasan.'
        + link + '</div></td></tr>'
        '</table></td></tr></table></body></html>')
    return subject, teks, html

AGENT_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "ambil_ulasan_baru",
            "description": "Gunakan alat ini PERTAMA KALI untuk mengambil ulasan terbaru dari Google Maps via Apify.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lihat_ulasan_negatif",
            "description": "Gunakan ini setelah mengambil ulasan. Untuk melihat berapa banyak ulasan negatif baru yang belum diemailkan.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kirim_email_agentic",
            "description": "Gunakan ini jika ada ulasan negatif. AI harus menganalisis teks dan menentukan urgensi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ringkasan": {"type": "string", "description": "1-3 kalimat santai tentang keluhan utama pasien."},
                    "urgensi": {"type": "string", "enum": ["TINGGI", "SEDANG", "RENDAH"], "description": "Tingkat bahaya keluhan. TINGGI jika menyangkut nyawa/kesalahan medis/amarah besar."},
                    "rekomendasi": {"type": "string", "description": "1 kalimat saran tindakan untuk manajemen RS."}
                },
                "required": ["ringkasan", "urgensi", "rekomendasi"]
            }
        }
    }
]

class OpsTools:
    def __init__(self, engine, preprocess, get_model, deploy,
                 fetch_items=None, send_mail=None, http_send=None):
        self.engine = engine
        self.preprocess = preprocess
        self.get_model = get_model
        self.deploy = deploy
        self.fetch_items = fetch_items or _apify_fetch
        self.send_mail = send_mail or _smtp_send
        self.http_send = http_send or _http_send
        self.schemas = []

    # ---- kv
    def _kv_get(self, k, default=None):
        with self.engine.connect() as c:
            v = c.execute(select(kv.c.v).where(kv.c.k == k)).scalar()
        return default if v is None else v

    def _kv_set(self, k, v):
        with self.engine.begin() as c:
            if not c.execute(update(kv).where(kv.c.k == k).values(v=str(v))).rowcount:
                c.execute(insert(kv).values(k=k, v=str(v)))

    def _canon(self, label):
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
        cutoff = (datetime.now(timezone.utc) - timedelta(days=NOTIFY_MAX_AGE_DAYS)) \
            .strftime("%Y-%m-%dT%H:%M:%S")
        with self.engine.begin() as c:
            c.execute(update(reviews).where(
                reviews.c.notified == 0,
                (reviews.c.published_at.is_(None)) | (reviews.c.published_at < cutoff)
            ).values(notified=1))

    def _since_date(self):
        with self.engine.connect() as c:
            mx = c.execute(select(func.max(reviews.c.published_at))).scalar()
        if not mx:
            return None
        try:
            return (datetime.fromisoformat(str(mx)[:10]) - timedelta(days=7)).strftime("%Y-%m-%d")
        except ValueError:
            return None

    def _simpan_info(self, places):
        now = _now()
        with self.engine.begin() as c:
            for p in places:
                nama = (p.get("title") or "").strip()
                if not nama:
                    continue
                cats = p.get("categories") or ([p["categoryName"]] if p.get("categoryName") else [])
                row = {"phone": p.get("phone") or p.get("phoneUnformatted"),
                       "website": p.get("website"),
                       "categories": json.dumps(cats, ensure_ascii=False),
                       "hours": json.dumps(p.get("openingHours") or [], ensure_ascii=False),
                       "extra": json.dumps(p.get("additionalInfo") or {}, ensure_ascii=False)[:20000],
                       "updated_at": now}
                if not c.execute(update(rs_info).where(rs_info.c.rs_name == nama)
                                 .values(**row)).rowcount:
                    c.execute(insert(rs_info).values(rs_name=nama, **row))

    # ---- 1. status
    def status_data(self):
        self._expire_old_pending()
        eff = self._eff()
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(reviews)).scalar()
            unl = c.execute(select(func.count()).select_from(reviews)
                            .where(reviews.c.label.is_(None))).scalar()
            pend = c.execute(select(func.count()).select_from(reviews)
                             .where(eff == "negatif", reviews.c.notified == 0)).scalar()
            corr = c.execute(select(func.count()).select_from(reviews)
                             .where(reviews.c.corrected_label.is_not(None))).scalar()
            by = c.execute(select(eff, func.count()).where(eff.is_not(None))
                           .group_by(eff)).all()
        _, _, metrics = self.get_model()
        last = int(self._kv_get("last_retrain_count", 0))
        return {"total_ulasan": total, "belum_berlabel": unl,
                "negatif_belum_diemail": pend,
                "per_label": {k: v for k, v in by}, "koreksi_total": corr,
                "koreksi_sejak_retrain_terakhir": corr - last,
                "syarat_retrain": MIN_NEW_CORRECTIONS,
                "f1_macro_model_aktif": metrics.get("f1_macro"),
                "retrain_terakhir": self._kv_get("last_retrain_result", "belum pernah")}

    # ---- 2. ambil ulasan
    def ambil_ulasan_baru(self, maks_per_rs=DEFAULT_MAX_REVIEWS):
        token = os.environ.get("APIFY_TOKEN")
        if not token:
            return {"error": "APIFY_TOKEN belum diset"}
        urls = self._place_urls()
        if not urls:
            return {"error": f"daftar RS kosong; isi {PLACES_FILE} dengan URL Google Maps tiap RS"}
        maks = _clamp(maks_per_rs, 1, 100, DEFAULT_MAX_REVIEWS)
        run_input = {"startUrls": [{"url": u} for u in urls], "language": "id",
                     "countryCode": "id", "maxCrawledPlacesPerSearch": 1,
                     "maxReviews": maks, "reviewsSort": "newest",
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
                if lat is None or lon is None or not (
                        bb["lat_min"] <= lat <= bb["lat_max"]
                        and bb["lng_min"] <= lon <= bb["lng_max"]):
                    luar += 1
                    continue
                rid = hashlib.sha256(
                    f"{nama}|{r.get('name')}|{r.get('publishedAtDate')}|{raw}".encode("utf-8")
                ).hexdigest()[:16]
                cand[rid] = {"review_id": rid, "rs_name": nama,
                             "username": r.get("name"), "rating": r.get("stars"),
                             "published_at": r.get("publishedAtDate"),
                             "address": place.get("address"),
                             "lat": lat, "lon": lon, "text": raw,
                             "scraped_at": _now(), "notified": 0, "legacy": 0}
        existing = set()
        with self.engine.connect() as c:
            for part in _chunks(list(cand), 500):
                existing.update(c.execute(select(reviews.c.review_id)
                                          .where(reviews.c.review_id.in_(part))).scalars())
        new = [v for k, v in cand.items() if k not in existing]
        if new:
            with self.engine.begin() as c:
                c.execute(insert(reviews), new)
        labeled = None
        if new:
            try:
                labeled = self.label_ulasan(batas=5000)
            except Exception as e:
                labeled = {"error": f"pelabelan gagal: {e}"}

        return {"dilabeli_otomatis": labeled, "ulasan_diterima": total,
                "baru": len(new), "sudah_ada": len(cand) - len(new),
                "tanpa_teks_dilewati": notext, "di_luar_indonesia_dibuang": luar,
                "sejak_tanggal": since,
                "baru_per_rs": dict(Counter(v["rs_name"] for v in new))}

    # ---- 3. label SVM + tag topik
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
                .values(label=bindparam("b_label"), confidence=bindparam("b_conf"),
                        clean=bindparam("b_clean"), processed_at=bindparam("b_pa"),
                        topics=bindparam("b_topics")))
        pa = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        params = []
        for r, l, cf, cl in zip(rows, labels, conf, clean):
            tps = sorted(topik_dari_teks(r.text or ""))
            params.append({"b_id": r.review_id, "b_label": l, "b_conf": cf,
                           "b_clean": cl, "b_pa": pa,
                           "b_topics": json.dumps(tps, ensure_ascii=False)})
        with self.engine.begin() as c:
            c.execute(stmt, params)
        return {"dilabeli": len(rows), "per_label": dict(Counter(labels)),
                "confidence_rata2": round(float(np.mean(conf)), 3)}

    # ---- 4. email ulasan negatif harian
    def _pending_negative(self, limit):
        self._expire_old_pending()
        with self.engine.connect() as c:
            return c.execute(select(reviews.c.review_id, reviews.c.rs_name, reviews.c.rating,
                                    reviews.c.text, reviews.c.published_at)
                             .where(self._eff() == "negatif", reviews.c.notified == 0)
                             .order_by(reviews.c.scraped_at).limit(limit)).all()

    def lihat_ulasan_negatif_belum_dikirim(self, batas=12):
        rows = self._pending_negative(_clamp(batas, 1, 30, 12))
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(reviews)
                              .where(self._eff() == "negatif",
                                     reviews.c.notified == 0)).scalar()
        return {"total_menunggu": total, "ditampilkan": len(rows),
                "ulasan": [{"rs": r.rs_name, "rating": r.rating,
                            "teks": (r.text or "")[:250]} for r in rows]}

    def _ringkasan_otomatis(self, rows):
        """Fallback ringkasan murni Python bila LLM tidak tersedia."""
        per_rs = {}
        for r in rows:
            tps = topik_dari_teks(r.text or "")
            per_rs.setdefault(r.rs_name, Counter()).update(tps or {"umum"})
        parts = []
        for rs, c in per_rs.items():
            top = ", ".join(f"{k} ({v}x)" for k, v in c.most_common(2))
            parts.append(f"{rs}: {top}")
        if not parts:
            return ""
        return "Keluhan utama — " + "; ".join(parts[:6]) + "."

    def kirim_email_negatif(self, ringkasan="", urgensi="RENDAH", rekomendasi=""):
        to = (os.environ.get("NOTIFY_TO") or "").strip()
        if not to:
            return {"error": "NOTIFY_TO belum diset"}
        provider = _email_provider()
        if provider == "smtp":
            user, pw = os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_APP_PASSWORD")
            if not (user and pw):
                return {"error": "GMAIL_USER / GMAIL_APP_PASSWORD belum diset "
                                 "(atau pakai RESEND_API_KEY / BREVO_API_KEY)"}
        elif provider == "resend" and not os.environ.get("RESEND_API_KEY"):
            return {"error": "RESEND_API_KEY belum diset"}
        elif provider == "brevo" and not (os.environ.get("BREVO_API_KEY")
                                          and os.environ.get("EMAIL_FROM")):
            return {"error": "BREVO_API_KEY dan EMAIL_FROM belum diset"}
        rows = self._pending_negative(50)
        if not rows:
            return {"dikirim": 0, "catatan": "tidak ada ulasan negatif baru"}
        if not (ringkasan or "").strip():
            ringkasan = self._ringkasan_otomatis(rows)
        recipients = [x.strip() for x in to.split(",") if x.strip()]
        subject, body, html = _susun_email(rows, ringkasan, urgensi, rekomendasi)
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
            return {"error": f"email gagal terkirim lewat {provider}: {e}{hint}; ulasan tetap antre"}
        with self.engine.begin() as c:
            c.execute(update(reviews)
                      .where(reviews.c.review_id.in_([r.review_id for r in rows]))
                      .values(notified=1))
        return {"dikirim": len(rows), "penerima": len(recipients), "lewat": provider,
                "urgensi": urgensi, "ringkasan_dipakai": bool(ringkasan)}

    # ---- 5. antrean ragu & koreksi
    def lihat_ulasan_ragu(self, batas=10):
        with self.engine.connect() as c:
            rows = c.execute(select(reviews.c.review_id, reviews.c.rs_name,
                                    reviews.c.rating, reviews.c.text,
                                    reviews.c.label, reviews.c.confidence)
                             .where(reviews.c.corrected_label.is_(None),
                                    reviews.c.label.is_not(None))
                             .order_by(reviews.c.confidence)
                             .limit(_clamp(batas, 1, 20, 10))).all()
        return {"jumlah": len(rows),
                "ulasan": [{"review_id": r.review_id, "rs": r.rs_name, "rating": r.rating,
                            "teks": (r.text or "")[:180], "label_svm": r.label,
                            "confidence": r.confidence} for r in rows]}

    def simpan_koreksi_manusia(self, review_id, label):
        lab = self._canon(label)
        if not lab:
            return {"error": "label bukan kelas valid"}
        with self.engine.begin() as c:
            n = c.execute(update(reviews).where(reviews.c.review_id == review_id)
                          .values(corrected_label=lab, label_source="manusia")).rowcount
        return {"tersimpan": n}

    # ---- 6. retrain
    def _read_labeled(self, path):
        import pandas as pd
        clean_path = path[:-4] + "_clean.csv" if path.endswith(".csv") else path
        src = clean_path if os.path.exists(clean_path) else path
        df = pd.read_csv(src)
        if TEXT_COL in df.columns:
            txt = df[TEXT_COL].astype(str)
        elif RAW_TEXT_COL in df.columns:
            txt = df[RAW_TEXT_COL].astype(str).map(self.preprocess)
        else:
            raise ValueError(f"{src}: butuh kolom '{TEXT_COL}' atau '{RAW_TEXT_COL}', "
                             f"ada {list(df.columns)}")
        out = pd.DataFrame({TEXT_COL: txt, LABEL_COL: df[LABEL_COL]}).dropna()
        out = out[out[TEXT_COL].str.strip().ne("") & out[TEXT_COL].ne("nan")]
        return out.reset_index(drop=True)

    def retrain_model(self):
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score)
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import LinearSVC

        with self.engine.connect() as c:
            corr = c.execute(select(reviews.c.clean, reviews.c.corrected_label)
                             .where(reviews.c.corrected_label.is_not(None),
                                    reviews.c.clean.is_not(None))).all()
        last = int(self._kv_get("last_retrain_count", 0))
        if len(corr) - last < MIN_NEW_CORRECTIONS:
            return {"dilewati": f"koreksi baru {len(corr) - last} < syarat {MIN_NEW_CORRECTIONS}"}
        if not (os.path.exists(TRAIN_POOL) and os.path.exists(TEST_SET)):
            return {"error": f"butuh {TRAIN_POOL} dan {TEST_SET}"}
        cur_pipe, cur_le, _ = self.get_model()
        canon = {str(c).lower(): str(c) for c in cur_le.classes_}
        norm = lambda s: canon.get(str(s).strip().lower(), str(s).strip())  # noqa: E731
        base = self._read_labeled(TRAIN_POOL)
        test = self._read_labeled(TEST_SET)
        extra = pd.DataFrame({TEXT_COL: [r.clean for r in corr],
                              LABEL_COL: [r.corrected_label for r in corr]})
        train = pd.concat([extra, base], ignore_index=True)
        train[LABEL_COL] = train[LABEL_COL].map(norm)
        test[LABEL_COL] = test[LABEL_COL].map(norm)
        train = train[~train[TEXT_COL].isin(set(test[TEXT_COL]))].drop_duplicates(TEXT_COL)
        le = LabelEncoder().fit(train[LABEL_COL])
        y = le.transform(train[LABEL_COL])
        min_cnt = min(Counter(y).values())
        if min_cnt < 2:
            return {"error": "salah satu kelas hanya punya 1 contoh; retrain dibatalkan"}
        grid = GridSearchCV(
            Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                      ("svm", LinearSVC(class_weight="balanced"))]),
            {"svm__C": [0.5, 1.0, 2.0]}, cv=max(2, min(5, min_cnt)),
            scoring="f1_macro", n_jobs=1)
        grid.fit(train[TEXT_COL], y)
        cand = grid.best_estimator_
        texts, truth = list(test[TEXT_COL]), list(test[LABEL_COL])
        cand_pred = list(le.inverse_transform(cand.predict(texts)))
        cur_pred = list(cur_le.inverse_transform(cur_pipe.predict(texts)))
        f1c = f1_score(truth, cand_pred, average="macro", zero_division=0)
        f1o = f1_score(truth, cur_pred, average="macro", zero_division=0)
        neg = canon.get("negatif", "Negatif")
        nr = lambda p: recall_score([t == neg for t in truth], [x == neg for x in p],  # noqa: E731
                                    zero_division=0)
        rc, ro = nr(cand_pred), nr(cur_pred)
        ok = f1c > f1o + MIN_F1_GAIN and rc >= ro - MAX_NEG_RECALL_DROP
        result = {"f1_model_lama": round(f1o, 4), "f1_kandidat": round(float(f1c), 4),
                  "recall_negatif_lama": round(ro, 4),
                  "recall_negatif_kandidat": round(rc, 4),
                  "contoh_latih": int(len(train)), "contoh_uji": len(truth),
                  "dipasang": bool(ok)}
        if ok:
            metrics = {"accuracy": round(accuracy_score(truth, cand_pred), 4),
                       "precision_macro": round(precision_score(truth, cand_pred,
                                                                average="macro",
                                                                zero_division=0), 4),
                       "recall_macro": round(recall_score(truth, cand_pred,
                                                          average="macro",
                                                          zero_division=0), 4),
                       "f1_macro": round(float(f1c), 4),
                       "cv_f1_macro": round(float(grid.best_score_), 4)}
            result["versi"] = save_model(self.engine, cand, le, metrics)
            self.deploy(cand, le, metrics)
            result["relabel"] = self._relabel_all()
        self._kv_set("last_retrain_count", len(corr))
        self._kv_set("last_retrain_result", json.dumps(result))
        return result

    def _relabel_all(self):
        return self.label_ulasan(batas=20000, semua=True).get("dilabeli", 0)

    # ---- 7. laporan bulanan
    @staticmethod
    def _period_bounds(period):
        y, m = period.split("-")
        start = f"{y}-{m}-01T00:00:00"
        if int(m) == 12:
            end = f"{int(y) + 1}-01-01T00:00:00"
        else:
            end = f"{y}-{int(m) + 1:02d}-01T00:00:00"
        return start, end

    @staticmethod
    def _shift_period(period, delta):
        y, m = map(int, period.split("-"))
        idx = y * 12 + (m - 1) + delta
        return f"{idx // 12:04d}-{idx % 12 + 1:02d}"

    def laporan_bulanan(self, periode=None):
        from datetime import datetime as _dt
        if not periode:
            now = _dt.utcnow()
            prev = now.replace(day=1) - timedelta(days=1)
            periode = prev.strftime("%Y-%m")
        try:
            start, end = self._period_bounds(periode)
        except Exception:
            return {"error": f"periode '{periode}' tidak valid; pakai YYYY-MM"}
        prev_period = self._shift_period(periode, -1)
        pstart, pend = self._period_bounds(prev_period)
        eff = self._eff()
        with self.engine.connect() as c:
            rows = c.execute(
                select(reviews.c.rs_name, reviews.c.rating, reviews.c.text,
                       eff.label("sent"))
                .where(reviews.c.published_at >= start, reviews.c.published_at < end,
                       eff.is_not(None))
            ).all()
            prev_rows = c.execute(
                select(reviews.c.rs_name, eff.label("sent"))
                .where(reviews.c.published_at >= pstart, reviews.c.published_at < pend,
                       eff.is_not(None))
            ).all()

        def agg(rs_rows):
            by = defaultdict(list)
            for r in rs_rows:
                by[r.rs_name].append(r)
            out = {}
            for rs, items in by.items():
                total = len(items)
                pos = sum(1 for x in items if x.sent == "positif")
                net = sum(1 for x in items if x.sent == "netral")
                neg = sum(1 for x in items if x.sent == "negatif")
                rats = [x.rating for x in items if x.rating is not None]
                avg = round(sum(rats) / len(rats), 2) if rats else None
                complaints, praises = Counter(), Counter()
                neg_samples = []
                for x in items:
                    tps = topik_dari_teks(x.text or "")
                    if x.sent == "negatif":
                        complaints.update(tps or {"umum"})
                        if x.text:
                            neg_samples.append(x.text)
                    elif x.sent == "positif":
                        praises.update(tps or {"umum"})
                neg_samples.sort(key=len)
                out[rs] = {"rs_name": rs, "period": periode, "total": total,
                           "positif": pos, "netral": net, "negatif": neg,
                           "avg_rating": avg,
                           "negatif_pct": round(100 * neg / total, 1) if total else 0.0,
                           "top_complaints": [{"topik": k, "count": v}
                                              for k, v in complaints.most_common(8)],
                           "top_praises": [{"topik": k, "count": v}
                                           for k, v in praises.most_common(8)],
                           "sample_negative": neg_samples[:3]}
            return out

        cur = agg(rows)
        prev = agg(prev_rows)
        now_iso = _now()
        saved = 0
        with self.engine.begin() as c:
            for rs, r in cur.items():
                prev_rs = prev.get(rs)
                delta = (r["negatif_pct"] - prev_rs["negatif_pct"]) if prev_rs else None
                payload = {
                    "rs_name": rs, "period": periode,
                    "total": r["total"], "positif": r["positif"],
                    "netral": r["netral"], "negatif": r["negatif"],
                    "avg_rating": r["avg_rating"],
                    "top_complaints": json.dumps(r["top_complaints"], ensure_ascii=False),
                    "top_praises": json.dumps(r["top_praises"], ensure_ascii=False),
                    "sample_negative": json.dumps(r["sample_negative"], ensure_ascii=False),
                    "prev_period": prev_period if prev_rs else None,
                    "delta_negatif_pct": delta, "generated_at": now_iso, "sent": 0,
                }
                existing = c.execute(select(monthly_reports.c.id)
                                     .where(monthly_reports.c.rs_name == rs,
                                            monthly_reports.c.period == periode)).scalar()
                if existing:
                    c.execute(update(monthly_reports)
                              .where(monthly_reports.c.id == existing)
                              .values(**{k: v for k, v in payload.items() if k != "sent"}))
                else:
                    c.execute(insert(monthly_reports).values(**payload))
                saved += 1
        return {"periode": periode, "jumlah_rs": saved,
                "total_ulasan": sum(r["total"] for r in cur.values()),
                "disimpan": saved}

    def kirim_laporan_bulanan(self, periode=None):
        to = (os.environ.get("MONTHLY_REPORT_TO")
              or os.environ.get("NOTIFY_TO") or "").strip()
        if not to:
            return {"error": "MONTHLY_REPORT_TO atau NOTIFY_TO belum diset"}
        if not periode:
            from datetime import datetime as _dt
            now = _dt.utcnow()
            prev = now.replace(day=1) - timedelta(days=1)
            periode = prev.strftime("%Y-%m")
        with self.engine.connect() as c:
            rows = c.execute(select(monthly_reports)
                             .where(monthly_reports.c.period == periode)
                             .order_by(monthly_reports.c.negatif.desc())).mappings().all()
        if not rows:
            return {"error": f"belum ada laporan untuk periode {periode}; jalankan laporan_bulanan dulu"}
        prev_period = self._shift_period(periode, -1)
        with self.engine.connect() as c:
            prev_rows_raw = c.execute(select(monthly_reports)
                                      .where(monthly_reports.c.period == prev_period)
                                      ).mappings().all()
        prev_map = {}
        for r in prev_rows_raw:
            prev_map[r["rs_name"]] = {
                "period": prev_period,
                "negatif_pct": round(100 * r["negatif"] / r["total"], 1) if r["total"] else 0.0,
            }
        data = []
        for r in rows:
            data.append({
                "rs_name": r["rs_name"], "total": r["total"],
                "positif": r["positif"], "netral": r["netral"], "negatif": r["negatif"],
                "avg_rating": r["avg_rating"],
                "top_complaints": json.loads(r["top_complaints"] or "[]"),
                "top_praises": json.loads(r["top_praises"] or "[]"),
                "sample_negative": json.loads(r["sample_negative"] or "[]"),
                "negatif_pct": round(100 * r["negatif"] / r["total"], 1) if r["total"] else 0.0,
            })
        subject, body, html = _susun_email_bulanan(periode, data, prev_map)
        provider = _email_provider()
        recipients = [x.strip() for x in to.split(",") if x.strip()]
        try:
            if provider == "smtp":
                user = os.environ.get("GMAIL_USER")
                pw = os.environ.get("GMAIL_APP_PASSWORD")
                if not (user and pw):
                    return {"error": "GMAIL_USER / GMAIL_APP_PASSWORD belum diset"}
                msg = EmailMessage()
                msg["Subject"], msg["From"], msg["To"] = subject, user, ", ".join(recipients)
                msg.set_content(body)
                msg.add_alternative(html, subtype="html")
                self.send_mail(user, pw, recipients, msg)
            else:
                self.http_send(provider, recipients, subject, body, html)
        except Exception as e:
            return {"error": f"laporan gagal dikirim lewat {provider}: {e}"}
        with self.engine.begin() as c:
            c.execute(update(monthly_reports)
                      .where(monthly_reports.c.period == periode)
                      .values(sent=1))
        return {"dikirim": len(rows), "periode": periode,
                "penerima": len(recipients), "lewat": provider}

    # ---- 8. deteksi lonjakan negatif (anomali)
    def cek_lonjakan_negatif(self):
        threshold = int(os.environ.get("ALERT_NEGATIF_THRESHOLD", "5"))
        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        cutoff_7d = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        eff = self._eff()
        with self.engine.connect() as c:
            recent = c.execute(select(reviews.c.rs_name, func.count())
                               .where(eff == "negatif",
                                      reviews.c.published_at >= cutoff_24h)
                               .group_by(reviews.c.rs_name)).all()
            base = c.execute(select(reviews.c.rs_name, func.count())
                             .where(eff == "negatif",
                                    reviews.c.published_at >= cutoff_7d,
                                    reviews.c.published_at < cutoff_24h)
                             .group_by(reviews.c.rs_name)).all()
        base_map = {rs: n / 6.0 for rs, n in base}
        lonjakan = []
        for rs, n in recent:
            avg = base_map.get(rs, 0.0)
            if n >= threshold and (avg == 0 or n >= 2 * avg):
                lonjakan.append({"rs": rs, "negatif_24h": int(n),
                                 "rata2_harian_7h": round(avg, 1)})
        if not lonjakan:
            return {"lonjakan": []}
        kirim = self._kirim_alert_lonjakan(lonjakan)
        return {"lonjakan": lonjakan, "email": kirim}

    def _kirim_alert_lonjakan(self, lonjakan):
        to = (os.environ.get("ALERT_TO") or os.environ.get("NOTIFY_TO") or "").strip()
        if not to:
            return {"skip": "ALERT_TO belum diset"}
        subject = f"[Geosentimen] LONJAKAN ulasan negatif di {len(lonjakan)} RS"
        lines = ["LONJAKAN ULASAN NEGATIF TERDETEKSI", ""]
        for x in lonjakan:
            lines.append(f"- {x['rs']}: {x['negatif_24h']} negatif dalam 24 jam "
                         f"(rata-rata harian 7 hari: {x['rata2_harian_7h']})")
        lines.append("\nSegera periksa dashboard untuk detail ulasan.")
        text = "\n".join(lines)
        html = ("<h2 style='color:#dc2626'>Lonjakan ulasan negatif</h2><ul>"
                + "".join(f"<li><b>{_html.escape(x['rs'])}</b>: {x['negatif_24h']} negatif "
                          f"dalam 24 jam (rata-rata {x['rata2_harian_7h']}/hari)</li>"
                          for x in lonjakan) + "</ul>")
        provider = _email_provider()
        recipients = [x.strip() for x in to.split(",") if x.strip()]
        try:
            if provider == "smtp":
                user = os.environ.get("GMAIL_USER")
                pw = os.environ.get("GMAIL_APP_PASSWORD")
                if not (user and pw):
                    return {"error": "SMTP belum diset"}
                msg = EmailMessage()
                msg["Subject"], msg["From"], msg["To"] = subject, user, ", ".join(recipients)
                msg.set_content(text)
                msg.add_alternative(html, subtype="html")
                self.send_mail(user, pw, recipients, msg)
            else:
                self.http_send(provider, recipients, subject, text, html)
            return {"dikirim": 1, "penerima": len(recipients)}
        except Exception as e:
            return {"error": str(e)}

    # ---- 9. dispatch
    def call(self, name, args):
        args = args if isinstance(args, dict) else {}
        fn = {
              "status_data": self.status_data,
              "ambil_ulasan_baru": self.ambil_ulasan_baru,
              "label_ulasan": self.label_ulasan,
              "lihat_ulasan_negatif_belum_dikirim": self.lihat_ulasan_negatif_belum_dikirim,
              "kirim_email_negatif": self.kirim_email_negatif,
              "kirim_email_agentic": self.kirim_email_negatif,
              "retrain_model": self.retrain_model,
              "laporan_bulanan": self.laporan_bulanan,
              "kirim_laporan_bulanan": self.kirim_laporan_bulanan,
              "cek_lonjakan_negatif": self.cek_lonjakan_negatif
              }.get(name)
        
        if not fn:
            return {"error": f"alat '{name}' tidak ada"}
        if name == "label_ulasan":
            args = {k: v for k, v in args.items() if k == "batas"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"argumen tidak valid: {e}"}
        except Exception as e:
            return {"error": f"alat gagal: {type(e).__name__}: {e}"}

    # ---- 10. health
    def _alert_ok(self, key):
        last = self._kv_get("alert_" + key)
        if last:
            try:
                if datetime.now(timezone.utc) - datetime.fromisoformat(
                        last.replace("Z", "+00:00")) < timedelta(hours=ALERT_COOLDOWN_H):
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
            self.http_send(prov, [x.strip() for x in to.split(",") if x.strip()],
                           f"[Geosentimen] PERINGATAN sistem: {jenis}", isi)
        except Exception as e:
            print("[health] gagal kirim peringatan:", e, flush=True)

    def _apify_usage(self):
        token = os.environ.get("APIFY_TOKEN")
        if not token:
            return None
        import httpx
        r = httpx.get("https://api.apify.com/v2/users/me/limits",
                      headers={"Authorization": f"Bearer {token}"}, timeout=15)
        if r.status_code != 200:
            return None
        d = r.json().get("data", {})
        used = d.get("current", {}).get("monthlyUsageUsd")
        cap = d.get("limits", {}).get("maxMonthlyUsageUsd")
        if used is None or not cap:
            return None
        return {"used": float(used), "cap": float(cap),
                "pct": round(100 * float(used) / float(cap), 1)}

    def pantau_kesehatan(self):
        masalah = []
        with self.engine.connect() as c:
            runs = c.execute(select(agent_runs.c.status, agent_runs.c.report)
                             .order_by(agent_runs.c.started_at.desc()).limit(2)).all()
        if len(runs) == 2 and all(r.status != "ok" for r in runs):
            masalah.append(("run_gagal",
                            "Run agen gagal 2 kali berturut-turut. Terakhir: "
                            + (runs[0].report or "")[:300]))
        try:
            u = self._apify_usage()
            if u and u["pct"] >= 80:
                masalah.append(("kredit_apify",
                                f"Kredit Apify terpakai {u['pct']}% "
                                f"(${u['used']:.2f} dari ${u['cap']:.2f})"))
        except Exception as e:
            print("[health] cek Apify gagal:", e, flush=True)
        try:
            rem = int(GROQ_SEEN.get("x-ratelimit-remaining-requests", ""))
            lim = int(GROQ_SEEN.get("x-ratelimit-limit-requests", ""))
            if lim and rem / lim < 0.1:
                masalah.append(("kuota_groq",
                                f"Kuota request harian Groq tinggal {rem} dari {lim}"))
        except ValueError:
            pass
        kirim = [m for m in masalah if self._alert_ok(m[0])]
        if kirim:
            self._kirim_peringatan(", ".join(k for k, _ in kirim),
                                   "\n\n".join(t for _, t in kirim))
        return [t for _, t in masalah]

    def status_kesehatan(self, maks_jam=13):
        with self.engine.connect() as c:
            ok = c.execute(select(agent_runs.c.finished_at)
                           .where(agent_runs.c.status == "ok")
                           .order_by(agent_runs.c.finished_at.desc()).limit(1)).scalar()
        umur = None
        if ok:
            umur = (datetime.now(timezone.utc)
                    - datetime.fromisoformat(ok.replace("Z", "+00:00"))).total_seconds() / 3600
        return {"ok": umur is not None and umur <= maks_jam,
                "jam_sejak_run_sukses": None if umur is None else round(umur, 1)}

_LOCK = asyncio.Lock()


def ops_running():
    return _LOCK.locked()


def _retry_wait(resp):
    try:
        ra = resp.headers.get("retry-after")
        if ra:
            return min(max(float(ra), 3.0), 70.0) + 1
    except Exception:
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", getattr(resp, "text", "") or "")
    if m:
        return min(max(int(m.group(1) or 0) * 60 + float(m.group(2)), 3.0), 70.0) + 1
    return 20.0


async def _post(client, url, headers, payload, tries=6):
    resp = None
    for _ in range(tries):
        resp = await client.post(url, headers=headers, json=payload)
        for k in ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                  "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens"):
            if resp.headers.get(k) is not None:
                GROQ_SEEN[k] = resp.headers.get(k)
        if resp.status_code != 429:
            return resp
        wait = _retry_wait(resp)
        print(f"[ops] kena batas Groq (429), menunggu {wait:.0f} detik", flush=True)
        await asyncio.sleep(wait)
    return resp

async def run_agentic_routine(tools, url, api_key, http_client=None):
    """Menjalankan rutinitas menggunakan FULL AGENTIC AI (Groq sebagai otak)."""
    if _LOCK.locked():
        return
    async with _LOCK:
        run_id = uuid.uuid4().hex[:12]
        with tools.engine.begin() as c:
            c.execute(update(agent_runs).where(agent_runs.c.status == "running").values(
                status="terputus", finished_at=_now(), report="Run terputus."))
            c.execute(insert(agent_runs).values(run_id=run_id, started_at=_now(), status="running"))
            
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        own_client = http_client is None
        if own_client:
            import httpx
            http_client = httpx.AsyncClient(timeout=120)
            
        messages = [
            {"role": "system", "content": "Kamu adalah Agen Cerdas Otonom untuk RS Banyumas. Tugasmu: 1) Panggil alat ambil_ulasan_baru. 2) Panggil lihat_ulasan_negatif. 3) Jika ada negatif, panggil kirim_email_agentic dengan analisis dan urgensimu sendiri. Jangan minta tolong manusia, lakukan semuanya otomatis."},
            {"role": "user", "content": "Jalankan rutinitas pemantauan ulasan sekarang."}
        ]
        
        log = []
        try:
            for step in range(5): # Maksimal 5 putaran berpikir
                payload = {
                    "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
                    "messages": messages,
                    "tools": AGENT_TOOLS_SCHEMA,
                    "tool_choice": "auto",
                    "temperature": 0.2
                }
                resp = await http_client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    raise Exception(f"Groq Error: {resp.text}")
                
                msg = resp.json()["choices"][0]["message"]
                messages.append(msg) # Simpan respons AI
                
                # Jika AI memutuskan memanggil alat (Function Calling)
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn_name = tc["function"]["name"]
                        args = json.loads(tc["function"]["arguments"])
                        print(f"[AGENTIC] AI memutuskan memanggil: {fn_name} dengan argumen {args}", flush=True)
                        
                        # Eksekusi alat Python berdasarkan perintah AI
                        if fn_name == "ambil_ulasan_baru":
                            res = await asyncio.to_thread(tools.call, "ambil_ulasan_baru", {})
                        elif fn_name == "lihat_ulasan_negatif":
                            res = await asyncio.to_thread(tools.call, "lihat_ulasan_negatif_belum_dikirim", {})
                        elif fn_name == "kirim_email_agentic":
                            res = await asyncio.to_thread(tools.call, "kirim_email_negatif", args)
                        else:
                            res = {"error": "alat tidak dikenal"}
                            
                        log.append({"alat": fn_name, "hasil": str(res)[:300]})
                        
                        # Kirim hasil alat kembali ke AI
                        messages.append({
                            "tool_call_id": tc["id"],
                            "role": "tool",
                            "name": fn_name,
                            "content": json.dumps(res, ensure_ascii=False)[:2000]
                        })
                else:
                    # AI selesai berpikir dan memberikan jawaban akhir
                    final_report = msg.get("content", "Selesai.")
                    print(f"[AGENTIC] Laporan Akhir AI: {final_report}", flush=True)
                    status = "ok"
                    report = final_report
                    break
            
            # Simpan hasil ke database
            with tools.engine.begin() as c:
                c.execute(update(agent_runs).where(agent_runs.c.run_id == run_id)
                          .values(finished_at=_now(), status=status, report=report[:4000],
                                  log=json.dumps(log, ensure_ascii=False)[:20000]))
        except Exception as e:
            with tools.engine.begin() as c:
                c.execute(update(agent_runs).where(agent_runs.c.run_id == run_id)
                          .values(finished_at=_now(), status="error", report=str(e)[:4000]))
        finally:
            if own_client and http_client:
                await http_client.aclose()