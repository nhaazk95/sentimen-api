import asyncio
import difflib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from types import SimpleNamespace

MAX_STEPS = 5          # batas putaran LLM per pertanyaan
TOOL_RESULT_CHARS = 6000
KEC_AMBIGU = {"banyumas"}   # nama kecamatan yang sama dengan kabupaten: hanya dipakai bila ditulis "kecamatan banyumas"
_SINGKATAN = [(r"\bsel\b\.?", "selatan"), (r"\but\b\.?", "utara"), (r"\btim\b\.?", "timur"), (r"\bbar\b\.?", "barat")]


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _words(text):
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split()


def _mentions(words, phrase, fuzzy=True):
    """True bila frasa ada di daftar kata; toleran salah ketik ringan."""
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


def _kecamatan(addr):
    """'... Kec. Purwokerto Sel., Kabupaten ...' -> 'Purwokerto Selatan' (singkatan arah dilengkapi)."""
    m = re.search(r"Kec\.?\s*([^,]+)", addr or "", flags=re.I)
    if not m:
        return ""
    k = m.group(1).strip()
    for pat, full in _SINGKATAN:
        k = re.sub(pat, full, k, flags=re.I)
    return " ".join(w.capitalize() for w in k.replace(".", " ").split())


def _mapbox_matrix(origin, dests):
    """Jarak dan waktu tempuh JALAN (mobil) dari satu titik ke banyak titik lewat Mapbox Matrix API.
    Butuh env MAPBOX_TOKEN di server (token terpisah dari token peta di browser). Mengembalikan daftar
    (meter, detik) searah dengan dests, atau None bila tidak tersedia."""
    token = os.environ.get("MAPBOX_TOKEN")
    if not token or not dests:
        return None
    import httpx
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in [origin] + list(dests))
    r = httpx.get(f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coords}", timeout=8,
                  params={"sources": "0", "destinations": ";".join(str(i) for i in range(1, len(dests) + 1)),
                          "annotations": "distance,duration", "access_token": token})
    if r.status_code != 200:
        return None
    j = r.json()
    if j.get("code") != "Ok":
        return None
    d, t = j["distances"][0], j["durations"][0]
    return [(d[i], t[i]) if d[i] is not None and t[i] is not None else None for i in range(len(dests))]


# Sinonim topik (dari buatKonteks di chat-widget-block.html), dipakai untuk memfilter ulasan per topik
TOPIK = {
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
    "fasilitas": ["fasilitas", "lengkap", "alat", "modern", "nyaman", "luas", "toilet", "musholla", "mushola", "kantin", "wifi", "lift"],
    "administrasi": ["administrasi", "pendaftaran", "daftar", "online", "loket", "kasir", "berkas", "registrasi", "rujukan", "faskes"],
    "jadwal": ["jadwal", "praktek", "praktik", "buka", "tutup", "libur"],
    "lokasi": ["lokasi", "strategis", "akses", "macet", "dekat", "jauh"],
    "informasi": ["informasi", "penjelasan", "komunikasi", "edukasi"],
}

AGENT_RULES = """ATURAN KEAKURATAN
1. Semua alamat, jarak, rating, jumlah ulasan, dan klaim layanan HANYA dari hasil alat. Jangan menebak atau mengarang. Bila alat tidak punya datanya, tulis "belum ada di data".
2. Untuk rekomendasi, lokasi, jarak, perbandingan, statistik, atau isi ulasan, panggil alat dulu. Boleh beberapa alat.
3. Jarak: salin angka dari alat, jangan menghitung sendiri. jarak_jalan_km dan perkiraan_menit = rute mobil tanpa kemacetan; jarak_garis_lurus_km = garis lurus. Selalu sebut jenisnya. Bila hanya garis lurus yang ada, katakan jarak jalan sebenarnya bisa lebih jauh dan sarankan cek Google Maps.
4. Pengguna berkata "dari sini", "lokasi saya", "dekat saya": panggil cari_rs_terdekat(lokasi="lokasi_pengguna"). Bila hasilnya "lokasi pengguna belum tersedia", minta pengguna menekan tombol 📍 di kepala chat dan mengizinkan lokasi, atau menyebut kecamatan atau RS asalnya. Jangan menebak lokasi.
5. Titik acuan kecamatan hanyalah perkiraan (rata-rata posisi RS di kecamatan itu). Sebut ini bila pengguna memakai nama kecamatan.
6. Hanya sebut RS yang ada di hasil alat, dengan nama persis seperti di data.
7. Ulasan Google Maps = sampel pengalaman pasien, bukan data resmi. Parafrasekan, tanpa nama pengguna. Kurang dari 20 ulasan: beri catatan "sampel kecil".
8. Jangan mendiagnosis atau menyarankan obat. Darurat (nyeri dada berat, sesak berat, tidak sadar, kejang, tanda stroke, perdarahan hebat): kalimat pertama sarankan IGD terdekat atau 112.
9. Di luar topik RS Banyumas: tolak sopan 1-2 kalimat. Sapaan: balas hangat singkat dengan 3 contoh pertanyaan.
10. Rekomendasi(lokasi, topik): peringkat gabungan jarak, sentimen topik, dan bukti ulasan yang dihitung sistem. Pakai untuk semua permintaan "rekomendasi RS untuk X", dengan atau tanpa lokasi. Pakai urutannya apa adanya dan jangan menyebut skor ke pengguna.
11. Bandingkan_rs(rs_a, rs_b): bandingkan dua RS. topik_keluhan(nama_rs): keluhan paling sering. tren_rs(nama_rs, bulan): perkembangan per bulan; ceritakan "kesimpulan" dari alat, jangan menghitung sendiri.
12. Info_rs(nama_rs): telepon, website, kategori, jam buka. Nomor telepon dan jam hanya boleh dari alat ini.
13. Bila ada jarak_jalan_km dan perkiraan_menit, sebut sebagai "rute mobil tanpa kemacetan"; selain itu pakai jarak garis lurus.
14. Tanpa lokasi dan bukan pertanyaan "terdekat" (mis. "rekomendasikan RS yang ada dokter giginya"): jangan bertanya lokasi. Panggil rekomendasi(topik=...) tanpa lokasi. Akhiri jawaban dengan satu kalimat saran untuk menyebut kecamatan atau memakai tombol lokasi.
15. Bila pertanyaan menyebut layanan atau topik (gigi, igd, anak, persalinan, dst), selalu pakai rekomendasi(lokasi, topik), walau ada kata "terdekat". cari_rs_terdekat hanya untuk pertanyaan jarak tanpa topik layanan. Bila lokasi pengguna di luar area data, katakan itu terus terang dan sebut RS terbaik untuk topik itu."""

FORMAT_RICH = """FORMAT: Markdown sesuai aturan format yang diberikan terpisah. Tiap RS wajib memakai nama persis seperti di hasil alat. Bila catatan alat memuat "di luar area data", kalimat pertama jawaban harus menyatakan bahwa lokasi pengguna berada di luar area data RS Banyumas."""

FORMAT_PLAIN = """FORMAT: teks biasa sesuai aturan format yang diberikan terpisah. Tiap RS wajib memakai nama persis seperti di hasil alat dan tiap sub-poin ditulis di baris sendiri. Bila catatan alat memuat "di luar area data", kalimat pertama jawaban harus menyatakan bahwa lokasi pengguna berada di luar area data RS Banyumas."""

def agent_system(fmt="rich"):
    """Prompt sistem ringkas untuk asisten chat (hemat token)."""
    return ("Kamu \"Asisten RS Banyumas\": pemandu informasi rumah sakit di Kabupaten Banyumas pada dashboard sentimen. "
            "Bahasa Indonesia, hangat, ringkas, dan akurat. Pakai \"saya\" untuk dirimu.\n\n"
            + AGENT_RULES + "\n\n" + (FORMAT_RICH if fmt == "rich" else FORMAT_PLAIN))


SCHEMAS = [
    {"type": "function", "function": {
        "name": "cari_rs_terdekat",
        "description": "RS terdekat dari satu titik asal, terurut dari yang terdekat, dengan alamat, jarak, rating. Titik asal: "
                       "lokasi pengguna (GPS), kecamatan, atau RS lain.",
        "parameters": {"type": "object", "properties": {
            "lokasi": {"type": "string", "description": "'lokasi_pengguna' untuk 'dari sini/dekat saya', atau nama kecamatan/kawasan, mis. 'Sokaraja'."},
            "dari_rs": {"type": "string", "description": "Nama RS asal, untuk 'RS terdekat dari RS X'. Pakai ini ATAU lokasi."},
            "topik": {"type": "string", "description": "Opsional: topik layanan (anak, igd, persalinan, ...) agar tiap RS menyertakan ringkasan ulasan topik itu."},
            "jumlah": {"type": "integer", "description": "Berapa RS (default 5, maks 8)."},
            "maks_km": {"type": "number", "description": "Batas jarak garis lurus (default 15)."}}}}},
    {"type": "function", "function": {
        "name": "jarak_antar_rs",
        "description": "Jarak dan alamat antara dua RS tertentu.",
        "parameters": {"type": "object", "properties": {
            "rs_a": {"type": "string"}, "rs_b": {"type": "string"}}, "required": ["rs_a", "rs_b"]}}},
    {"type": "function", "function": {
        "name": "statistik_rs",
        "description": "Profil satu RS: alamat, rating, jumlah ulasan, sebaran sentimen.",
        "parameters": {"type": "object", "properties": {
            "nama_rs": {"type": "string", "description": "Nama RS, boleh sebagian."}}, "required": ["nama_rs"]}}},
    {"type": "function", "function": {
        "name": "ranking_rs",
        "description": "Peringkat RS dari ulasan yang menyebut suatu topik (skor dihaluskan agar adil untuk RS berulasan sedikit).",
        "parameters": {"type": "object", "properties": {
            "topik": {"type": "string", "description": "mis. igd, anak, persalinan, antrean, farmasi, kebersihan, parkir, bpjs."},
            "urut": {"type": "string", "enum": ["positif", "keluhan_sedikit", "rating", "jumlah_ulasan"]},
            "jumlah": {"type": "integer"}, "min_ulasan": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "cari_ulasan",
        "description": "Kutipan ulasan pasien yang relevan dengan kata kunci; opsional per RS atau sentimen.",
        "parameters": {"type": "object", "properties": {
            "kata_kunci": {"type": "string"}, "nama_rs": {"type": "string"},
            "sentimen": {"type": "string", "enum": ["positif", "netral", "negatif"]},
            "jumlah": {"type": "integer"}}, "required": ["kata_kunci"]}}},
]


def merge_system(messages):
    """Gabungkan semua pesan system menjadi satu di awal (beberapa penyedia, mis. Gemini, menolak banyak pesan system)."""
    parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    rest = [m for m in messages if m.get("role") != "system"]
    return ([{"role": "system", "content": "\n\n".join(parts)}] if parts else []) + rest


class AgentError(Exception):
    def __init__(self, status, detail=""):
        super().__init__(f"{status}: {str(detail)[:200]}")
        self.status = status


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


class HospitalTools:
    def __init__(self, hospitals, reviews, rev_index, preprocess, pipeline, label_encoder,
                 stop, find_origin=None, haversine=None, road_fn=None):
        """pipeline/label_encoder boleh None bila tiap review sudah punya r["sent"] (mode database).
        find_origin dan haversine hanya untuk kompatibilitas; titik asal kini dihitung di dalam kelas ini.
        road_fn(origin, dests) -> [(meter, detik)] untuk jarak jalan; False = matikan; None = Mapbox bila ada token."""
        self.hospitals = hospitals
        for h in self.hospitals:
            h["kec"] = _kecamatan(h.get("alamat")) or h.get("kec", "")
        self.by_name = {h["nama"]: h for h in hospitals}
        self.reviews = reviews
        self.rev_index = rev_index
        self.n_rev = len(reviews)
        self.preprocess = preprocess
        self.stop = set(stop)
        self.haversine = _haversine
        self.road_fn = road_fn
        self.schemas = SCHEMAS
        if pipeline is not None:
            self._label_all(pipeline, label_encoder)
        self.by_rs = defaultdict(list)
        for r in self.reviews:
            self.by_rs[r["rs"]].append(r)
        self.kec_known = sorted({h["kec"] for h in hospitals if h.get("kec")})
        self._build_geo()

    # Label sentimen dihitung sekali memakai model SVM aktif (mode xlsx lama).
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
        """Titik tengah tiap kecamatan = rata-rata posisi RS di kecamatan itu (perkiraan, bukan batas wilayah resmi)."""
        pts = defaultdict(list)
        for h in self.hospitals:
            if h.get("kec") and h["lat"] is not None:
                pts[h["kec"].lower()].append((h["lat"], h["lon"]))
        self._kec_centroid = {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)) for k, v in pts.items()}
        pwt = [c for k, c in self._kec_centroid.items() if k.startswith("purwokerto")]
        self._kota_alias = ({"purwokerto": (sum(a for a, _ in pwt) / len(pwt), sum(b for _, b in pwt) / len(pwt))}
                            if pwt else {})

    def _origin_dari_teks(self, text):
        words = _words(text)
        for kec in sorted(self._kec_centroid, key=len, reverse=True):
            k = " ".join(_words(kec))
            if kec in KEC_AMBIGU:
                hit = _mentions(words, f"kecamatan {k}", fuzzy=False) or _mentions(words, f"kec {k}", fuzzy=False)
            else:
                hit = _mentions(words, k)
            if hit:
                c = self._kec_centroid[kec]
                return c[0], c[1], f"Kec. {kec.title()} (titik perkiraan dari posisi RS di kecamatan itu)"
        for name, c in self._kota_alias.items():
            if _mentions(words, name):
                return c[0], c[1], "pusat kota Purwokerto (titik perkiraan dari posisi RS di kota)"
        return None

    def _road(self, origin, hospitals):
        """[(meter, detik) atau None] searah hospitals, atau None bila jarak jalan tidak tersedia."""
        if self.road_fn is False:
            return None
        fn = self.road_fn or _mapbox_matrix
        try:
            res = fn((origin[0], origin[1]), [(h["lat"], h["lon"]) for h in hospitals])
        except Exception:
            return None
        return res if res and len(res) == len(hospitals) else None

    # ------------------------------------------------------------------ util
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
            if t == name or name in words or any(w == s or (len(s) >= 5 and s in w) for w in words for s in syn):
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

    # ----------------------------------------------------------------- alat
    def cari_rs_terdekat(self, lokasi=None, dari_rs=None, jumlah=5, maks_km=15, topik=None, user_latlng=None):
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
            origin = (h0["lat"], h0["lon"], f"{nama} ({h0['alamat'] or 'alamat belum ada di data'})")
            exclude = nama
        else:
            teks = (lokasi or "").strip()
            if not teks or re.search(r"pengguna|saya|sini|gps|lokasiku", teks, re.I):
                ok = (user_latlng and user_latlng[0] is not None and user_latlng[1] is not None
                      and -90 <= user_latlng[0] <= 90 and -180 <= user_latlng[1] <= 180)
                if not ok:
                    return {"error": "lokasi pengguna belum tersedia (pengguna belum mengizinkan GPS)",
                            "saran": "minta pengguna menekan tombol 📍 di kepala chat dan mengizinkan lokasi, "
                                     "atau menyebut nama kecamatan atau RS asal"}
                origin = (user_latlng[0], user_latlng[1], "lokasi pengguna (GPS)")
            else:
                origin = self._origin_dari_teks(teks)
                if not origin:
                    return {"error": f"lokasi '{lokasi}' tidak dikenali", "kecamatan_dikenal": self.kec_known}

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
            catatan.append(f"tidak ada RS dalam {maks_km:g} km; ini {len(short)} yang paling dekat")
        road = self._road(origin, [h for _, h in short])
        items = []
        for i, (d, h) in enumerate(short):
            it = {"nama": h["nama"], "alamat": h["alamat"] or "belum ada di data", "kecamatan": h["kec"],
                  "jarak_garis_lurus_km": round(d, 1),
                  "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                  "jumlah_ulasan": h["n"]}
            if road and road[i]:
                it["jarak_jalan_km"] = round(road[i][0] / 1000, 1)
                it["perkiraan_menit"] = round(road[i][1] / 60)
            if topik:
                rx = self._topic_regex(topik)
                rows = [r for r in self.by_rs.get(h["nama"], []) if rx and rx.search(r["text"])]
                a = self._agg(rows)
                it["ulasan_topik"] = {"topik": topik, "jumlah": a["n"], "positif_pct": _pct(a["positif"], a["n"]),
                                      "negatif_pct": _pct(a["negatif"], a["n"])}
                if a["n"] < 10:
                    it["ulasan_topik"]["catatan"] = "sampel kecil"
            items.append(it)
        if road:
            items.sort(key=lambda x: x.get("jarak_jalan_km", 1e9))
            catatan.append("jarak_jalan_km dan perkiraan_menit = rute mobil tanpa kemacetan; jarak_garis_lurus_km = garis lurus")
        else:
            catatan.append("hanya jarak garis lurus yang tersedia; jarak jalan sebenarnya bisa lebih jauh")
        out = {"titik_acuan": origin[2], "catatan": "; ".join(catatan), "rs": items[:jumlah]}
        if tanpa:
            out["rs_tanpa_koordinat_tidak_dihitung"] = tanpa
        return out

    def jarak_antar_rs(self, rs_a="", rs_b=""):
        a, err = self._one_rs(rs_a)
        if err:
            return err
        b, err = self._one_rs(rs_b)
        if err:
            return err
        ha, hb = self.by_name[a], self.by_name[b]
        if ha["lat"] is None or hb["lat"] is None:
            return {"error": "koordinat salah satu RS belum ada di data"}
        out = {"dari": {"nama": a, "alamat": ha["alamat"] or "belum ada di data"},
               "ke": {"nama": b, "alamat": hb["alamat"] or "belum ada di data"},
               "jarak_garis_lurus_km": round(_haversine(ha["lat"], ha["lon"], hb["lat"], hb["lon"]), 1)}
        road = self._road((ha["lat"], ha["lon"]), [hb])
        if road and road[0]:
            out["jarak_jalan_km"] = round(road[0][0] / 1000, 1)
            out["perkiraan_menit"] = round(road[0][1] / 60)
            out["catatan"] = "jarak_jalan_km dan perkiraan_menit = rute mobil tanpa kemacetan"
        else:
            out["catatan"] = "hanya jarak garis lurus yang tersedia; jarak jalan sebenarnya bisa lebih jauh"
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
        m = 10   # bobot prior: RS dengan ulasan sedikit ditarik ke rata-rata
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
                                "positif_pct": _pct(a["positif"], n), "negatif_pct": _pct(a["negatif"], n),
                                "rating_rata2": a["rating"],
                                "sampel_kecil": n < 20}))
        if not items:
            return {"error": f"tidak ada RS dengan minimal {min_ulasan} ulasan yang menyebut '{topik}'",
                    "total_ulasan_cocok": N, "saran": "turunkan min_ulasan"}
        items.sort(key=lambda x: x[0])
        return {"topik": topik or "(semua ulasan)", "urut": urut, "total_ulasan_cocok": N,
                "catatan": "urutan memakai skor yang dihaluskan; persen di bawah adalah angka mentah",
                "peringkat": [dict(no=i + 1, **it) for i, (_, it) in enumerate(items[:jumlah])]}

    def cari_ulasan(self, kata_kunci="", nama_rs=None, sentimen=None, jumlah=6):
        jumlah = _clamp(jumlah, 1, 10, 6)
        only = None
        if nama_rs:
            only, err = self._one_rs(nama_rs)
            if err:
                return err
        want = _norm_sent(sentimen)
        toks = {t for t in self.preprocess(kata_kunci).split()
                if len(t) >= 3 and t not in self.stop and t in self.rev_index
                and len(self.rev_index[t]) <= 0.4 * self.n_rev}
        if not toks:
            return {"error": "kata kunci terlalu umum atau tidak ada di ulasan", "kata_kunci": kata_kunci}
        score = Counter()
        for t in toks:
            w = math.log(1 + self.n_rev / len(self.rev_index[t]))
            for i in self.rev_index[t]:
                score[i] += w
        cap = jumlah if only else 2
        per, out = Counter(), []
        for i, _ in sorted(score.items(), key=lambda kv: (-kv[1], len(self.reviews[kv[0]]["text"]))):
            rv = self.reviews[i]
            if only and rv["rs"] != only:
                continue
            if want and rv["sent"] != want:
                continue
            if per[rv["rs"]] >= cap:
                continue
            per[rv["rs"]] += 1
            txt = rv["text"]
            if len(txt) > 220:
                txt = txt[:220].rsplit(" ", 1)[0] + "..."
            out.append({"rs": rv["rs"], "rating": rv["rating"], "sentimen": rv["sent"], "kutipan": txt})
            if len(out) >= jumlah:
                break
        if not out:
            return {"error": "tidak ada ulasan yang cocok dengan filter", "kata_kunci": kata_kunci}
        return {"hasil": out, "catatan": "kutipan tanpa nama pengguna; parafrasekan saat menjawab"}

    # --------------------------------------------------------------- dispatch
    def call(self, name, args, user_latlng=None):
        args = args if isinstance(args, dict) else {}
        try:
            if name == "cari_rs_terdekat":
                return self.cari_rs_terdekat(user_latlng=user_latlng, **args)
            if name == "jarak_antar_rs":
                return self.jarak_antar_rs(**args)
            if name == "statistik_rs":
                return self.statistik_rs(**args)
            if name == "ranking_rs":
                return self.ranking_rs(**args)
            if name == "cari_ulasan":
                return self.cari_ulasan(**args)
            return {"error": f"alat '{name}' tidak ada"}
        except TypeError as e:           # argumen salah nama/jenis: model boleh mencoba lagi
            return {"error": f"argumen tidak valid: {e}"}
        except Exception as e:           # jangan sampai satu alat rusak menjatuhkan seluruh /chat
            return {"error": f"alat gagal: {e}"}


def _retry_wait(resp):
    """Lama menunggu setelah 429: header Retry-After atau teks 'try again in 1m6.1s'."""
    try:
        ra = resp.headers.get("retry-after")
        if ra:
            return min(max(float(ra), 2.0), 30.0) + 1
    except Exception:
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", getattr(resp, "text", "") or "")
    if m:
        return min(max(int(m.group(1) or 0) * 60 + float(m.group(2)), 2.0), 30.0) + 1
    return 8.0


async def _post(client, url, headers, payload, budget):
    """POST ke LLM. Bila kena batas token per menit (429), tunggu lalu ulangi selama anggaran waktu (detik) masih ada,
    supaya pengguna melihat 'mengetik' sedikit lebih lama, bukan pesan error."""
    while True:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 429:
            return resp
        wait = _retry_wait(resp)
        if budget["sisa"] < wait:
            return resp
        budget["sisa"] -= wait
        print(f"[agent] kena batas LLM (429), menunggu {wait:.0f} detik", flush=True)
        await asyncio.sleep(wait)


async def run_agent(client, url, headers, base_payload, messages, tools, user_latlng=None, wait_budget=40):
    """Putaran agen. Mengembalikan teks jawaban akhir (\"\" bila model tidak menghasilkan teks)."""
    msgs = merge_system(messages)
    if user_latlng and user_latlng[0] is not None and user_latlng[1] is not None:
        hint = ("\n\nINFO SISTEM: lokasi GPS pengguna SUDAH tersedia. Untuk 'terdekat', 'dekat saya', 'dari saya', "
                "atau 'dari sini', langsung panggil cari_rs_terdekat(lokasi=\"lokasi_pengguna\") atau "
                "rekomendasi(lokasi=\"lokasi_pengguna\", topik=...). Jangan meminta kecamatan.")
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = dict(msgs[0], content=msgs[0]["content"] + hint)
        else:
            msgs.insert(0, {"role": "system", "content": hint.strip()})
    budget = {"sisa": float(wait_budget)}
    tool_fails, nudged = 0, False
    for step in range(MAX_STEPS):
        payload = dict(base_payload, messages=msgs, tools=tools.schemas)
        resp = await _post(client, url, headers, payload, budget)
        if resp.status_code == 400 and "tool_use_failed" in (resp.text or "") and tool_fails < 1:
            tool_fails += 1                      # model menulis panggilan alat yang rusak: minta ulang sekali
            msgs.append({"role": "user", "content": "Panggilan alat sebelumnya rusak. Ulangi dengan nama alat yang lengkap dan benar."})
            continue
        if resp.status_code != 200:
            raise AgentError(resp.status_code, resp.text)
        msg = resp.json().get("choices", [{}])[0].get("message", {})
        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()
        if not calls:
            if not content and step > 0 and not nudged:
                nudged = True                    # setelah memakai alat model diam: minta jawaban akhir
                msgs.append({"role": "user", "content": "Tulis jawaban akhir untuk pengguna berdasarkan hasil alat di atas."})
                continue
            return content
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await asyncio.to_thread(tools.call, fn.get("name"), args, user_latlng)
            print(f"[agent] {fn.get('name')}({args}) -> {str(result)[:120]}", flush=True)
            msgs.append({"role": "tool", "tool_call_id": c.get("id"), "name": fn.get("name"),
                         "content": json.dumps(result, ensure_ascii=False)[:TOOL_RESULT_CHARS]})
    return ""