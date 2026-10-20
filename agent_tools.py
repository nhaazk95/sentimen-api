import json, math, time, urllib.request, difflib
from collections import Counter
from datetime import datetime, timedelta
from fastapi import APIRouter

router = APIRouter()

REVIEWS_URL = "https://raw.githubusercontent.com/<nhaazk95>/dashboard_rs_banyumas/main/docs/data/reviews.json"
PROFIL = json.load(open("data/rs_profil.json", encoding="utf-8"))
_cache = {"t": 0, "data": []}

KATA = {
    "persalinan": ["persalinan", "lahiran", "bersalin", "bidan", "caesar", "kandungan", "hamil", "usg"],
    "igd": ["igd", "ugd", "darurat"],
    "farmasi": ["farmasi", "obat", "apotek", "resep"],
    "antrean": ["antre", "antri", "tunggu", "nunggu"],
    "bpjs": ["bpjs", "jkn", "kis"],
    "gigi": ["gigi"],
    "anak": ["anak", "bayi", "nicu", "balita"],
    "bedah": ["bedah", "operasi"],
    "radiologi": ["radiologi", "rontgen", "ct scan", "mri"],
    "parkir": ["parkir"],
    "kebersihan": ["bersih", "kotor", "bau"],
}

def semua_ulasan():
    if time.time() - _cache["t"] > 3600:  # segarkan tiap 1 jam
        with urllib.request.urlopen(REVIEWS_URL, timeout=20) as f:
            _cache["data"] = json.load(f)
        _cache["t"] = time.time()
    return _cache["data"]

def wib(r):
    t = datetime.fromisoformat(r["Waktu Ulasan"].replace("Z", "+00:00"))
    return t + timedelta(hours=7)

def ringkas(r):  # tanpa Username
    return {"rs": r["Nama RS"], "rating": r.get("Rating"),
            "tanggal": wib(r).strftime("%d %b %Y"), "jam_wib": wib(r).hour,
            "label": r.get("Sentimen_Prediksi"), "teks": r.get("Isi Ulasan")}

def cocok_rs(r, rs):
    return not rs or rs.lower() in r["Nama RS"].lower()

def jarak(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * 6371 * math.asin(math.sqrt(h))

def titik_daerah(daerah):
    d = daerah.lower()
    pts = [(p["lat"], p["lng"]) for p in PROFIL
           if p["lat"] and p["kecamatan"] and d in p["kecamatan"].lower()]
    if not pts:
        pts = [(p["lat"], p["lng"]) for p in PROFIL
               if p["lat"] and d in (p["nama"] + " " + (p["alamat"] or "")).lower()]
    if not pts:
        return None
    return (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))

@router.get("/rs-terdekat")
def rs_terdekat(daerah: str = None, lat: float = None, lng: float = None):
    titik = (lat, lng) if lat is not None and lng is not None else (
        titik_daerah(daerah) if daerah else None)
    if not titik:
        return {"error": "lokasi tidak dikenali, minta pengguna menyebut kecamatan"}
    hasil = sorted(
        [{"nama": p["nama"], "alamat": p["alamat"], "kecamatan": p["kecamatan"],
          "jarak_km": round(jarak(titik, (p["lat"], p["lng"])), 1),
          "rating": p["rating"], "jumlah_ulasan": p["jumlah_ulasan"]}
         for p in PROFIL if p["lat"]],
        key=lambda x: x["jarak_km"])
    for h in hasil:
        h["lebih_dari_15_km"] = h["jarak_km"] > 15
    return {"titik_acuan": daerah or "lokasi pengguna",
            "catatan": "jarak garis lurus (perkiraan)",
            "peringatan": "lokasi di luar Kabupaten Banyumas" if hasil[0]["jarak_km"] > 30 else None,
            "hasil": hasil[:5]}

@router.get("/profil-rs")
def profil_rs(nama: str):
    nama_l = nama.lower()
    for p in PROFIL:
        if nama_l in p["nama"].lower():
            return p
    m = difflib.get_close_matches(nama, [p["nama"] for p in PROFIL], n=1, cutoff=0.5)
    return next((p for p in PROFIL if m and p["nama"] == m[0]), {"error": "RS tidak ditemukan"})

@router.get("/ulasan-topik")
def ulasan_topik(topik: str, rs: str = None):
    kunci = KATA.get(topik.lower(), [topik.lower()])
    out = [r for r in semua_ulasan() if cocok_rs(r, rs)
           and any(k in (r.get("Isi Ulasan") or "").lower() for k in kunci)]
    return {"topik": topik, "jumlah": len(out), "kutipan": [ringkas(r) for r in out[:8]]}

@router.get("/ulasan-per-jam")
def ulasan_per_jam(jam: int, rs: str = None):
    data = [r for r in semua_ulasan() if cocok_rs(r, rs)]
    per_jam = Counter(wib(r).hour for r in data)
    if not per_jam:
        return {"jumlah": 0, "catatan": "tidak ada data"}
    dipakai = jam if per_jam.get(jam) else min(per_jam, key=lambda h: abs(h - jam))
    out = [r for r in data if wib(r).hour == dipakai]
    n = len(out)
    per_label = Counter(r.get("Sentimen_Prediksi") for r in out)
    return {
        "jam_diminta": jam, "jam_dipakai": dipakai,
        "tepat": dipakai == jam, "selisih_jam": abs(dipakai - jam),
        "jumlah": n,
        "per_label": {l: per_label.get(l, 0) for l in ("Positif", "Netral", "Negatif")},
        "persen": {l: round(per_label.get(l, 0) * 100 / n, 1) for l in ("Positif", "Netral", "Negatif")},
        "rs_terbanyak": Counter(r["Nama RS"] for r in out).most_common(3),
        "contoh": [ringkas(r) for r in out[:5]],
        "sebaran_jam_wib": dict(sorted(per_jam.items())),
        "catatan": "jam dari stempel waktu (UTC dikonversi ke WIB), bisa perkiraan",
    }