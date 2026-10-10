"""
agent.py -- mode agen untuk endpoint /chat (tool calling lewat Groq, API kompatibel OpenAI).

Letakkan di root repo sentimen-api, sejajar dengan main.py. Modul ini tidak meng-import main.py
(supaya tidak circular); semua data dan fungsi dari main.py dioper lewat HospitalTools(...).

Alur: LLM menerima pertanyaan + daftar alat -> memutuskan alat apa yang dipanggil -> hasil alat
dikembalikan ke LLM -> diulang sampai LLM menjawab (maksimal MAX_STEPS putaran).
"""
import difflib
import json
import math
import re
from collections import Counter, defaultdict
from types import SimpleNamespace

MAX_STEPS = 5          # putaran terakhir memaksa model menjawab tanpa alat
TOOL_RESULT_CHARS = 6000

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
}

AGENT_RULES = """# Alat (mode agen)
Kamu punya alat untuk membaca data ulasan RS Banyumas. Untuk pertanyaan tentang rekomendasi, peringkat, perbandingan, jarak, statistik, atau isi ulasan, WAJIB panggil alat dulu; jangan menjawab dari ingatan.
- cari_rs_terdekat(lokasi): RS terdekat dari kecamatan atau kawasan, lengkap dengan jarak garis lurus.
- statistik_rs(nama_rs): profil, rating, dan sebaran sentimen satu RS.
- ranking_rs(topik, urut): peringkat RS berdasarkan sentimen ulasan yang menyebut suatu topik (antrean, IGD, anak, persalinan, dan sebagainya).
- cari_ulasan(kata_kunci, nama_rs, sentimen): kutipan ulasan yang relevan.
Aturan memakai alat:
- Kamu boleh memanggil beberapa alat, berurutan atau sekaligus. Contoh: "RS terdekat dari Sokaraja untuk anak demam" = cari_rs_terdekat lalu cari_ulasan atau ranking_rs dengan topik "anak" untuk memeriksa RS yang dekat.
- Hasil alat setara dengan PROFIL RS, DATA DARI DASHBOARD, dan KUTIPAN ULASAN pada aturan di atas. Alamat, jarak, rating, dan klaim layanan hanya boleh berasal dari hasil alat. Kalau hasil alat berisi error atau kosong, katakan terus terang dan jangan mengarang.
- Persentase sentimen dari alat adalah hasil model SVM pada ulasan sampel. Sebut jumlah ulasannya, dan beri catatan "sampel kecil" bila kurang dari 20 ulasan.
- Sapaan atau pertanyaan umum yang tidak butuh data: jawab langsung tanpa alat."""

SCHEMAS = [
    {"type": "function", "function": {
        "name": "cari_rs_terdekat",
        "description": "Cari RS terdekat dari suatu kecamatan/kawasan di Banyumas (jarak garis lurus, terurut dari yang terdekat).",
        "parameters": {"type": "object", "properties": {
            "lokasi": {"type": "string",
                       "description": "Nama kecamatan atau kawasan, mis. 'Sokaraja' atau 'Purwokerto Selatan'. Isi 'lokasi_pengguna' bila pengguna minta 'dekat saya'."},
            "jumlah": {"type": "integer", "description": "Berapa RS (default 5, maks 8)."},
            "maks_km": {"type": "number", "description": "Batas jarak dalam km (default 15)."}},
            "required": ["lokasi"]}}},
    {"type": "function", "function": {
        "name": "statistik_rs",
        "description": "Profil satu RS: alamat, kecamatan, rating rata-rata, jumlah ulasan, dan sebaran sentimen (positif/netral/negatif).",
        "parameters": {"type": "object", "properties": {
            "nama_rs": {"type": "string", "description": "Nama RS, boleh sebagian, mis. 'Hermina'."}},
            "required": ["nama_rs"]}}},
    {"type": "function", "function": {
        "name": "ranking_rs",
        "description": "Peringkat RS dari ulasan yang menyebut suatu topik. Skor sudah dihaluskan (Bayesian) supaya RS dengan ulasan sedikit tidak menang secara tidak adil.",
        "parameters": {"type": "object", "properties": {
            "topik": {"type": "string",
                      "description": "Topik/layanan, mis. 'igd', 'anak', 'persalinan', 'antrean', 'farmasi', 'kebersihan', 'parkir', 'bpjs'. Kosongkan untuk semua ulasan."},
            "urut": {"type": "string", "enum": ["positif", "keluhan_sedikit", "rating", "jumlah_ulasan"],
                     "description": "Kriteria urutan (default 'positif')."},
            "jumlah": {"type": "integer", "description": "Berapa RS (default 5, maks 8)."},
            "min_ulasan": {"type": "integer", "description": "Minimal ulasan yang cocok per RS (default 5)."}}}}},
    {"type": "function", "function": {
        "name": "cari_ulasan",
        "description": "Cari kutipan ulasan pasien yang relevan dengan kata kunci, opsional untuk satu RS atau satu jenis sentimen.",
        "parameters": {"type": "object", "properties": {
            "kata_kunci": {"type": "string", "description": "Kata kunci atau kalimat pendek, mis. 'dokter anak ramah'."},
            "nama_rs": {"type": "string", "description": "Opsional: batasi ke satu RS."},
            "sentimen": {"type": "string", "enum": ["positif", "netral", "negatif"], "description": "Opsional."},
            "jumlah": {"type": "integer", "description": "Berapa kutipan (default 6, maks 10)."}},
            "required": ["kata_kunci"]}}},
]


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
                 stop, find_origin, haversine):
        self.hospitals = hospitals
        self.by_name = {h["nama"]: h for h in hospitals}
        self.reviews = reviews
        self.rev_index = rev_index
        self.n_rev = len(reviews)
        self.preprocess = preprocess
        self.stop = set(stop)
        self.find_origin = find_origin
        self.haversine = haversine
        self.schemas = SCHEMAS
        self._label_all(pipeline, label_encoder)
        self.by_rs = defaultdict(list)
        for r in self.reviews:
            self.by_rs[r["rs"]].append(r)
        self.kec_known = sorted({h["kec"] for h in hospitals if h.get("kec")})

    # Label sentimen dihitung sekali saat start memakai model SVM yang sedang dipakai.
    # Setelah retrain + redeploy, label otomatis ikut model baru.
    def _label_all(self, pipeline, label_encoder):
        for r in self.reviews:
            r["sent"] = None
        texts = [r.get("clean") or " ".join(sorted(r["tok"])) for r in self.reviews]
        if not texts:
            return
        labels = label_encoder.inverse_transform(pipeline.predict(texts))
        for r, lab in zip(self.reviews, labels):
            r["sent"] = _norm_sent(lab)

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
    def cari_rs_terdekat(self, lokasi="", jumlah=5, maks_km=15, user_latlng=None):
        jumlah = _clamp(jumlah, 1, 8, 5)
        try:
            maks_km = float(maks_km)
        except (TypeError, ValueError):
            maks_km = 15.0
        req = SimpleNamespace(lat=None, lng=None, message=lokasi or "", history=[])
        if (not lokasi or "pengguna" in str(lokasi).lower()) and user_latlng and user_latlng[0] is not None:
            req.lat, req.lng = user_latlng
        origin = self.find_origin(req)
        if not origin:
            return {"error": f"lokasi '{lokasi}' tidak dikenali (butuh nama kecamatan atau Purwokerto)",
                    "kecamatan_dikenal": self.kec_known}
        ranked = []
        for h in self.hospitals:
            if h["lat"] is None:
                continue
            ranked.append((self.haversine(origin[0], origin[1], h["lat"], h["lon"]), h))
        ranked.sort(key=lambda x: x[0])
        near = [x for x in ranked if x[0] <= maks_km][:jumlah]
        catatan = "jarak garis lurus perkiraan, bukan jarak rute"
        if not near:
            near = ranked[:3]
            catatan += f"; tidak ada RS dalam {maks_km:g} km, ini 3 yang paling dekat"
        return {"titik_acuan": origin[2], "catatan": catatan,
                "rs": [{"nama": h["nama"], "alamat": h["alamat"], "kecamatan": h["kec"],
                        "jarak_km": round(d, 1),
                        "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                        "jumlah_ulasan": h["n"]} for d, h in near]}

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


async def run_agent(client, url, headers, base_payload, messages, tools, user_latlng=None):
    """Putaran agen. Mengembalikan teks jawaban akhir ("" bila model tidak menghasilkan teks)."""
    msgs = list(messages)
    for step in range(MAX_STEPS):
        payload = dict(base_payload, messages=msgs, tools=tools.schemas,
                       tool_choice="none" if step == MAX_STEPS - 1 else "auto")
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise AgentError(resp.status_code, resp.text)
        msg = resp.json().get("choices", [{}])[0].get("message", {})
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip()
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = tools.call(fn.get("name"), args, user_latlng)
            print(f"[agent] {fn.get('name')}({args}) -> {str(result)[:120]}", flush=True)
            msgs.append({"role": "tool", "tool_call_id": c.get("id"),
                         "content": json.dumps(result, ensure_ascii=False)[:TOOL_RESULT_CHARS]})
    return ""
