"""agent.py — utilitas bersama untuk agen operasional Geosentimen RS Banyumas.

Modul ini TIDAK lagi dipakai untuk chatbot pengguna. Yang tersisa:
- Haversine, helper teks (kata, frasa, kecamatan)
- _mapbox_matrix (jarak jalan)
- TOPIK (sinonim topik, dipakai untuk tag & laporan bulanan)
- HospitalTools (tetap dipakai ops_agent untuk analisis ulasan / topik / jarak)
"""

import difflib
import math
import os
import re
from collections import Counter, defaultdict

try:
    from types import SimpleNamespace  # noqa: F401
except Exception:
    pass


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _words(text):
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split()


def _mentions(words, phrase, fuzzy=True):
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


_SINGKATAN = [(r"\bsel\b\.?", "selatan"), (r"\but\b\.?", "utara"),
              (r"\btim\b\.?", "timur"), (r"\bbar\b\.?", "barat")]


def _kecamatan(addr):
    m = re.search(r"Kec\.?\s*([^,]+)", addr or "", flags=re.I)
    if not m:
        return ""
    k = m.group(1).strip()
    for pat, full in _SINGKATAN:
        k = re.sub(pat, full, k, flags=re.I)
    return " ".join(w.capitalize() for w in k.replace(".", " ").split())


def _mapbox_matrix(origin, dests):
    token = os.environ.get("MAPBOX_TOKEN")
    if not token or not dests:
        return None
    import httpx
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in [origin] + list(dests))
    r = httpx.get(
        f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coords}", timeout=8,
        params={"sources": "0",
                "destinations": ";".join(str(i) for i in range(1, len(dests) + 1)),
                "annotations": "distance,duration", "access_token": token})
    if r.status_code != 200:
        return None
    j = r.json()
    if j.get("code") != "Ok":
        return None
    d, t = j["distances"][0], j["durations"][0]
    return [(d[i], t[i]) if d[i] is not None and t[i] is not None else None for i in range(len(dests))]


# Sinonim topik: dipakai untuk tag ulasan dan agregasi laporan bulanan.
TOPIK = {
    "persalinan": ["persalinan", "lahiran", "melahirkan", "bersalin", "bidan", "caesar", "sesar",
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
    "dokter": ["dokter", "dr"],
    "perawat": ["perawat", "suster", "nurse"],
    "inap": ["inap", "ranap", "opname", "kamar", "bangsal", "vip", "icu", "hcu", "kelas"],
    "poli": ["poli", "poliklinik", "spesialis", "konsultasi", "kontrol"],
    "jantung": ["jantung", "kardiologi", "kardio", "ekg", "ecg", "stent", "kateterisasi"],
    "saraf": ["saraf", "neurologi", "stroke", "epilepsi", "vertigo"],
    "tulang": ["tulang", "ortopedi", "orthopaedi", "ortho", "patah", "fraktur", "sendi"],
    "fisioterapi": ["fisioterapi", "fisio", "rehabilitasi", "rehab", "terapi"],
    "mata": ["mata", "katarak", "kacamata", "lasik"],
    "tht": ["tht", "telinga", "hidung", "tenggorokan", "amandel"],
    "kulit": ["kulit", "dermatologi", "jerawat", "kelamin"],
    "paru": ["paru", "tbc", "asma", "batuk", "sesak", "pneumonia"],
    "ginjal": ["ginjal", "dialisis", "hemodialisa", "urologi"],
    "diabetes": ["diabetes", "gula", "insulin"],
    "hipertensi": ["hipertensi", "tensi", "kolesterol"],
    "jiwa": ["jiwa", "psikolog", "psikiater", "psikologi", "depresi", "konseling"],
    "kanker": ["kanker", "tumor", "onkologi", "kemoterapi", "kemo", "radioterapi", "biopsi"],
    "dental": ["gigi", "behel", "kawat", "scaling", "tambal", "cabut", "geraham", "ortodonti"],
    "sunat": ["sunat", "khitan", "sirkumsisi"],
    "kb": ["kb", "spiral", "implan", "kontrasepsi"],
    "gizi": ["gizi", "stunting", "laktasi", "asi", "nutrisi"],
    "vaksin": ["vaksin", "vaksinasi", "imunisasi", "suntik"],
    "covid": ["covid", "corona", "isolasi", "swab", "pcr", "antigen"],
    "lab": ["lab", "laboratorium", "urin", "darah", "patologi"],
    "mcu": ["mcu", "checkup", "skrining", "screening"],
    "ambulans": ["ambulans", "ambulance", "evakuasi"],
    "biaya": ["biaya", "harga", "mahal", "murah", "tarif", "bayar", "tagihan", "deposit", "kwitansi"],
    "keramahan": ["ramah", "sopan", "galak", "judes", "cuek", "kasar", "sabar", "senyum"],
    "kecepatan": ["cepat", "lambat", "lama", "tanggap", "sigap", "responsif", "telat"],
    "fasilitas": ["fasilitas", "lengkap", "alat", "modern", "nyaman", "luas", "toilet",
                  "musholla", "mushola", "kantin", "wifi", "lift"],
    "administrasi": ["administrasi", "pendaftaran", "daftar", "online", "loket", "kasir",
                     "berkas", "registrasi", "rujukan", "faskes"],
    "jadwal": ["jadwal", "praktek", "praktik", "buka", "tutup", "libur"],
    "lokasi": ["lokasi", "strategis", "akses", "macet", "dekat", "jauh"],
    "informasi": ["informasi", "penjelasan", "komunikasi", "edukasi"],
}


def topik_dari_teks(text):
    """Kembalikan set nama topik yang disebut dalam satu teks ulasan."""
    t = (text or "").lower()
    hit = set()
    for nama, syn in TOPIK.items():
        for s in syn:
            if len(s) >= 5:
                if s in t:
                    hit.add(nama)
                    break
            else:
                if re.search(r"\b" + re.escape(s) + r"\b", t):
                    hit.add(nama)
                    break
    return hit


class HospitalTools:
    """Alat analisis RS/ulasan. Dipakai OpsTools untuk laporan bulanan & pemeriksaan,
    bukan lagi untuk chatbot pengguna.
    """

    def __init__(self, hospitals, reviews, rev_index, preprocess,
                 label_encoder=None, pipeline=None, stop=None,
                 road_fn=None):
        self.hospitals = hospitals
        for h in self.hospitals:
            h["kec"] = _kecamatan(h.get("alamat")) or h.get("kec", "")
        self.by_name = {h["nama"]: h for h in hospitals}
        self.reviews = reviews
        self.rev_index = rev_index
        self.n_rev = len(reviews)
        self.preprocess = preprocess
        self.stop = set(stop or ())
        self.road_fn = road_fn
        if pipeline is not None and label_encoder is not None:
            self._label_all(pipeline, label_encoder)
        self.by_rs = defaultdict(list)
        for r in self.reviews:
            self.by_rs[r["rs"]].append(r)
        self.kec_known = sorted({h["kec"] for h in hospitals if h.get("kec")})
        self._build_geo()

    def _label_all(self, pipeline, label_encoder):
        for r in self.reviews:
            r["sent"] = None
        texts = [r.get("clean") or " ".join(sorted(r["tok"])) for r in self.reviews]
        if not texts:
            return
        labels = label_encoder.inverse_transform(pipeline.predict(texts))
        for r, lab in zip(self.reviews, labels):
            r["sent"] = _norm_sent(lab)

    def _build_geo(self):
        pts = defaultdict(list)
        for h in self.hospitals:
            if h.get("kec") and h["lat"] is not None:
                pts[h["kec"].lower()].append((h["lat"], h["lon"]))
        self._kec_centroid = {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v))
                              for k, v in pts.items()}
        pwt = [c for k, c in self._kec_centroid.items() if k.startswith("purwokerto")]
        self._kota_alias = ({"purwokerto": (sum(a for a, _ in pwt) / len(pwt),
                                            sum(b for _, b in pwt) / len(pwt))} if pwt else {})

    def _origin_dari_teks(self, text):
        words = _words(text)
        for kec in sorted(self._kec_centroid, key=len, reverse=True):
            k = " ".join(_words(kec))
            if kec == "banyumas":
                hit = (_mentions(words, f"kecamatan {k}", fuzzy=False)
                       or _mentions(words, f"kec {k}", fuzzy=False))
            else:
                hit = _mentions(words, k)
            if hit:
                c = self._kec_centroid[kec]
                return c[0], c[1], f"Kec. {kec.title()}"
        for name, c in self._kota_alias.items():
            if _mentions(words, name):
                return c[0], c[1], "pusat kota Purwokerto"
        return None

    def _road(self, origin, hospitals):
        if self.road_fn is False:
            return None
        fn = self.road_fn or _mapbox_matrix
        try:
            res = fn((origin[0], origin[1]), [(h["lat"], h["lon"]) for h in hospitals])
        except Exception:
            return None
        return res if res and len(res) == len(hospitals) else None

    def _find_rs(self, q):
        q = (q or "").strip().lower()
        if not q:
            return []
        low = {n.lower(): n for n in self.by_name}
        if q in low:
            return [low[q]]
        sub = [low[k] for k in low if q in k or k in q]
        if sub:
            return sub
        words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) >= 3 and w not in self.stop]
        if words:
            hit = [low[k] for k in low if all(w in k for w in words)]
            if hit:
                return hit
        return [low[m] for m in difflib.get_close_matches(q, list(low), n=3, cutoff=0.6)]

    def _one_rs(self, q):
        cand = self._find_rs(q)
        if not cand:
            return None, {"error": f"RS '{q}' tidak ditemukan", "daftar_rs": sorted(self.by_name)[:40]}
        if len(cand) > 1:
            return None, {"error": "nama RS ambigu, pilih salah satu", "kandidat": cand[:8]}
        return cand[0], None

    @staticmethod
    def _topic_regex(topik):
        t = (topik or "").lower().strip()
        if not t:
            return None
        words = [w for w in re.split(r"[^a-z0-9]+", t) if len(w) >= 3]
        keys = set()
        for name, syn in TOPIK.items():
            if t == name or name in words or any(
                    w == s or (len(s) >= 5 and s in w) for w in words for s in syn):
                keys.update(syn)
                keys.add(name)
        if not keys:
            keys.update(words)
        if not keys:
            return None
        pat = [re.escape(k) if len(k) >= 5 else r"\b" + re.escape(k) + r"\b" for k in keys]
        return re.compile("(" + "|".join(pat) + ")", re.I)

    @staticmethod
    def _agg(rows):
        c = Counter(r["sent"] for r in rows)
        n = len(rows)
        rats = [r["rating"] for r in rows if r.get("rating") is not None]
        return {"n": n, "positif": c.get("positif", 0), "netral": c.get("netral", 0),
                "negatif": c.get("negatif", 0),
                "rating": round(sum(rats) / len(rats), 2) if rats else None}

    def cari_rs_terdekat(self, lokasi=None, dari_rs=None, jumlah=5, maks_km=15,
                         topik=None, user_latlng=None):
        jumlah = _clamp(jumlah, 1, 8, 5)
        try:
            maks_km = float(maks_km)
        except (TypeError, ValueError):
            maks_km = 15.0
        exclude = None
        if dari_rs:
            nama, err = self._one_rs(dari_rs)
            if err:
                return err
            h0 = self.by_name[nama]
            if h0["lat"] is None:
                return {"error": f"koordinat {nama} belum ada di data"}
            origin = (h0["lat"], h0["lon"], f"{nama} ({h0['alamat'] or 'alamat belum ada'})")
            exclude = nama
        else:
            teks = (lokasi or "").strip()
            if not teks or re.search(r"pengguna|saya|sini|gps|lokasiku", teks, re.I):
                ok = (user_latlng and user_latlng[0] is not None and user_latlng[1] is not None
                      and -90 <= user_latlng[0] <= 90 and -180 <= user_latlng[1] <= 180)
                if not ok:
                    return {"error": "lokasi pengguna belum tersedia"}
                origin = (user_latlng[0], user_latlng[1], "lokasi pengguna (GPS)")
            else:
                origin = self._origin_dari_teks(teks)
                if not origin:
                    return {"error": f"lokasi '{lokasi}' tidak dikenali",
                            "kecamatan_dikenal": self.kec_known}
        cand, tanpa = [], []
        for h in self.hospitals:
            if h["nama"] == exclude:
                continue
            if h["lat"] is None or h["lon"] is None:
                tanpa.append(h["nama"])
                continue
            cand.append((_haversine(origin[0], origin[1], h["lat"], h["lon"]), h))
        cand.sort(key=lambda x: x[0])
        short = [x for x in cand if x[0] <= maks_km][:8]
        catatan = []
        if not short:
            short = cand[:3]
            catatan.append(f"tidak ada RS dalam {maks_km:g} km")
        road = self._road(origin, [h for _, h in short])
        items = []
        for i, (d, h) in enumerate(short):
            it = {"nama": h["nama"], "alamat": h["alamat"] or "belum ada di data",
                  "kecamatan": h["kec"], "jarak_garis_lurus_km": round(d, 1),
                  "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                  "jumlah_ulasan": h["n"]}
            if road and road[i]:
                it["jarak_jalan_km"] = round(road[i][0] / 1000, 1)
                it["perkiraan_menit"] = round(road[i][1] / 60)
            if topik:
                rx = self._topic_regex(topik)
                rows = [r for r in self.by_rs.get(h["nama"], []) if rx and rx.search(r["text"])]
                a = self._agg(rows)
                it["ulasan_topik"] = {"topik": topik, "jumlah": a["n"],
                                      "positif_pct": _pct(a["positif"], a["n"]),
                                      "negatif_pct": _pct(a["negatif"], a["n"])}
                if a["n"] < 10:
                    it["ulasan_topik"]["catatan"] = "sampel kecil"
            items.append(it)
        if road:
            items.sort(key=lambda x: x.get("jarak_jalan_km", 1e9))
            catatan.append("jarak_jalan_km & perkiraan_menit = rute mobil tanpa kemacetan")
        else:
            catatan.append("hanya jarak garis lurus yang tersedia")
        out = {"titik_acuan": origin[2], "catatan": "; ".join(catatan), "rs": items[:jumlah]}
        if tanpa:
            out["rs_tanpa_koordinat_tidak_dihitung"] = tanpa
        return out

    def statistik_rs(self, nama_rs=""):
        nama, err = self._one_rs(nama_rs)
        if err:
            return err
        h = self.by_name[nama]
        a = self._agg(self.by_rs.get(nama, []))
        out = {"nama": nama, "alamat": h["alamat"] or "belum ada di data", "kecamatan": h["kec"],
               "rating_rata2_semua_ulasan": round(h["rating"], 2) if h["rating"] is not None else None,
               "jumlah_ulasan_semua": h["n"],
               "analisis_sentimen": {"jumlah_ulasan_teks": a["n"],
                                     "positif_pct": _pct(a["positif"], a["n"]),
                                     "netral_pct": _pct(a["netral"], a["n"]),
                                     "negatif_pct": _pct(a["negatif"], a["n"])}}
        if a["n"] < 20:
            out["catatan"] = "sampel kecil (kurang dari 20 ulasan teks)"
        return out

    def ranking_rs(self, topik=None, urut="positif", jumlah=5, min_ulasan=5):
        jumlah = _clamp(jumlah, 1, 8, 5)
        min_ulasan = _clamp(min_ulasan, 1, 200, 5)
        if urut not in ("positif", "keluhan_sedikit", "rating", "jumlah_ulasan"):
            urut = "positif"
        rx = self._topic_regex(topik)
        by = defaultdict(list)
        for r in self.reviews:
            if r["sent"] and (rx is None or rx.search(r["text"])):
                by[r["rs"]].append(r)
        allrows = [r for rows in by.values() for r in rows]
        if not allrows:
            return {"error": f"tidak ada ulasan yang menyebut topik '{topik}'"}
        N = len(allrows)
        p0 = sum(r["sent"] == "positif" for r in allrows) / N
        q0 = sum(r["sent"] == "negatif" for r in allrows) / N
        rats = [r["rating"] for r in allrows if r.get("rating") is not None]
        mu = sum(rats) / len(rats) if rats else 0.0
        m = 10
        items = []
        for rs, rows in by.items():
            a = self._agg(rows)
            n = a["n"]
            if n < min_ulasan:
                continue
            nr = [r["rating"] for r in rows if r.get("rating") is not None]
            sp = (a["positif"] + m * p0) / (n + m)
            sn = (a["negatif"] + m * q0) / (n + m)
            sr = (sum(nr) + m * mu) / (len(nr) + m) if nr else mu
            key = {"positif": -sp, "keluhan_sedikit": sn, "rating": -sr, "jumlah_ulasan": -n}[urut]
            items.append((key, {"rs": rs, "ulasan_cocok": n,
                                "positif_pct": _pct(a["positif"], n),
                                "negatif_pct": _pct(a["negatif"], n),
                                "rating_rata2": a["rating"], "sampel_kecil": n < 20}))
        if not items:
            return {"error": f"tidak ada RS dengan minimal {min_ulasan} ulasan topik '{topik}'",
                    "total_ulasan_cocok": N}
        items.sort(key=lambda x: x[0])
        return {"topik": topik or "(semua ulasan)", "urut": urut,
                "total_ulasan_cocok": N, "peringkat": [dict(no=i + 1, **it)
                                                        for i, (_, it) in enumerate(items[:jumlah])]}


def _norm_sent(s):
    s = str(s or "").strip().lower()
    if s.startswith("pos"):
        return "positif"
    if s.startswith("neg"):
        return "negatif"
    if s.startswith("net"):
        return "netral"
    return None


def _clamp(v, lo, hi, default):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _pct(x, n):
    return round(100.0 * x / n, 1) if n else None