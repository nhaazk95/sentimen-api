"""tools_ext.py -- alat tambahan agen chat: rekomendasi gabungan, bandingkan, keluhan, tren, info layanan.
Mewarisi HospitalTools dari agent.py. Jarak jalan memakai self._road() milik agent.py (Mapbox), bukan OSRM."""
import contextvars
import re
from collections import Counter, defaultdict

from agent import HospitalTools, TOPIK

CALLS = contextvars.ContextVar("calls", default=None)   # catatan alat yang dipanggil (untuk log pertanyaan)
TOPIK_EXTRA = {"biaya": ["biaya", "mahal", "tarif", "bayar"],
               "keramahan": ["ramah", "judes", "galak", "cuek"],
               "fasilitas": ["fasilitas", "kamar", "ruang tunggu", "toilet"]}


def _cl(v, lo, hi, d):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return d
    return max(lo, min(hi, v))


def _pc(x, n):
    return round(100.0 * x / n, 1) if n else None

def _key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def _fn(name, desc, props, req=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req or []}}}


NM = {"type": "string", "description": "Nama RS, boleh sebagian."}
EXTRA_SCHEMAS = [
    _fn("rekomendasi",
        "Peringkat RS gabungan: jarak (bila lokasi ada), sentimen ulasan yang menyebut topik, dan bukti ulasan. "
        "Skor dihitung sistem. Boleh tanpa lokasi.",
        {"lokasi": {"type": "string", "description": "Kecamatan/kawasan, atau 'lokasi_pengguna' bila minta dekat saya. Kosongkan bila pengguna tidak menyebut lokasi."},
         "topik": {"type": "string", "description": "Layanan/topik, mis. gigi, igd, anak, persalinan."},
         "maks_km": {"type": "number"}, "jumlah": {"type": "integer"}}),
    _fn("bandingkan_rs", "Bandingkan dua RS berdampingan: rating, sentimen, keluhan utama.",
        {"rs_a": NM, "rs_b": NM, "topik": {"type": "string"}}, ["rs_a", "rs_b"]),
    _fn("topik_keluhan", "Keluhan paling sering di satu RS (dari ulasan negatif) beserta contoh.",
        {"nama_rs": NM, "jumlah": {"type": "integer"}}, ["nama_rs"]),
    _fn("tren_rs", "Perkembangan rating dan sentimen satu RS per bulan, beserta kesimpulan dari sistem.",
        {"nama_rs": NM, "bulan": {"type": "integer"}}, ["nama_rs"]),
    _fn("info_rs", "Telepon, website, kategori, dan jam buka satu RS dari Google Maps.", {"nama_rs": NM}, ["nama_rs"]),
]


class ToolsPlus(HospitalTools):
    def __init__(self, *a, info=None, **kw):
        super().__init__(*a, **kw)
        self.info = info or {}
        self.schemas = list(self.schemas) + EXTRA_SCHEMAS
        self._rx = {k: self._topic_regex(k) for k in TOPIK}
        for k, ws in TOPIK_EXTRA.items():
            pat = [re.escape(w) if len(w) >= 5 else r"\b" + re.escape(w) + r"\b" for w in ws]
            self._rx[k] = re.compile("(" + "|".join(pat) + ")", re.I)

    # ---- util
    def _origin(self, lokasi, user_latlng):
        """(lat, lon, label) dari GPS atau teks kecamatan; None bila tidak ada/tidak dikenali."""
        teks = (lokasi or "").strip()
        gps = bool(user_latlng and user_latlng[0] is not None and user_latlng[1] is not None)
        if gps and (not teks or re.search(r"pengguna|saya|sini|gps|lokasiku", teks, re.I)):
            return (user_latlng[0], user_latlng[1], "lokasi pengguna (GPS)")
        return self._origin_dari_teks(teks) if teks else None

    def _keluhan(self, nama, k=5):
        cnt, contoh = Counter(), {}
        for r in self.by_rs.get(nama, []):
            if r["sent"] != "negatif":
                continue
            for t, rx in self._rx.items():
                if rx and rx.search(r["text"]):
                    cnt[t] += 1
                    if t not in contoh and 30 <= len(r["text"]) <= 220:
                        contoh[t] = r["text"]
        return [{"topik": t, "jumlah": n, "contoh": contoh.get(t)} for t, n in cnt.most_common(k)]

    # ---- alat
    def rekomendasi(self, lokasi="", topik=None, maks_km=15, jumlah=5, user_latlng=None):
        jumlah = _cl(jumlah, 1, 8, 5)
        try:
            maks_km = float(maks_km)
        except (TypeError, ValueError):
            maks_km = 15.0
        origin = self._origin(lokasi, user_latlng) if (lokasi or user_latlng) else None
        if lokasi and not origin:
            return {"error": f"lokasi '{lokasi}' tidak dikenali atau lokasi pengguna belum tersedia",
                    "saran": "minta pengguna menekan tombol 📍 atau menyebut kecamatan",
                    "kecamatan_dikenal": self.kec_known}
        rx = self._topic_regex(topik) if topik else None
        by = defaultdict(list)
        for r in self.reviews:
            if r["sent"] and (rx is None or rx.search(r["text"])):
                by[r["rs"]].append(r)
        allr = [r for rows in by.values() for r in rows]
        if topik and not allr:
            return {"error": f"tidak ada ulasan yang menyebut topik '{topik}'"}
        p0 = sum(r["sent"] == "positif" for r in allr) / (len(allr) or 1)
        m, items = 10, []
        for h in self.hospitals:
            rows = by.get(h["nama"], [])
            n, pos = len(rows), sum(r["sent"] == "positif" for r in rows)
            d = self.haversine(origin[0], origin[1], h["lat"], h["lon"]) if origin and h["lat"] is not None else None
            items.append({"h": h, "n": n, "sp": (pos + m * p0) / (n + m), "d": d, "pos": _pc(pos, n)})
        far = False
        catatan = "skor: sentimen topik dan bukti ulasan (tanpa jarak)"
        if origin:
            near = [x for x in items if x["d"] is not None and x["d"] <= maks_km]
            if near:
                items = near
                catatan = "skor: jarak 45%, sentimen topik 35%, bukti ulasan 20%"
            else:
                far = True
                catatan = (f"lokasi pengguna di luar area data (tidak ada RS dalam {maks_km:g} km); "
                           "peringkat memakai sentimen topik tanpa jarak, jarak tetap ditampilkan sebagai informasi")
        w = (0.45, 0.35, 0.20) if (origin and not far) else (0.0, 0.65, 0.35)
        for x in items:     
            dist = max(0.0, 1 - x["d"] / maks_km) if x["d"] is not None else 0.0
            cov = min(1.0, x["n"] / 10) if topik else min(1.0, (x["h"]["n"] or 0) / 100)
            x["skor"] = round(w[0] * dist + w[1] * x["sp"] + w[2] * cov, 3)
        items.sort(key=lambda x: -x["skor"])
        top = items[:jumlah]

        road = self._road(origin, [x["h"] for x in top]) if origin else None
        out = []
        for i, x in enumerate(top):
            h = x["h"]
            e = {"nama": h["nama"], "alamat": (h["alamat"] or "belum ada di data")[:90], "kecamatan": h["kec"],
                 "skor": x["skor"],
                 "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                 "positif_pct": x["pos"], "ulasan_menyebut_topik": x["n"],
                 "bukti_layanan": (x["n"] > 0) if topik else None}
            if x["d"] is not None:
                e["jarak_garis_lurus_km"] = round(x["d"], 1)
            if road and road[i]:
                e["jarak_jalan_km"] = round(road[i][0] / 1000, 1)
                e["perkiraan_menit"] = round(road[i][1] / 60)
            cand = sorted((r for r in by.get(h["nama"], [])
                           if r["sent"] == "positif" and 25 <= len(r["text"]) <= 200),
                          key=lambda r: len(r["text"]))
            if topik and cand:
                e["contoh_ulasan"] = cand[0]["text"][:120]
            out.append(e)
        if origin:
            catatan += ("; jarak_jalan_km dan perkiraan_menit = rute mobil tanpa kemacetan" if road
                        else "; hanya jarak garis lurus yang tersedia, jarak jalan sebenarnya bisa lebih jauh")
        return {"titik_acuan": origin[2] if origin else "tanpa lokasi", "topik": topik or "(umum)",
                "catatan": catatan, "rs": out}

    def bandingkan_rs(self, rs_a="", rs_b="", topik=None):
        hasil = []
        rx = self._topic_regex(topik) if topik else None
        for q in (rs_a, rs_b):
            nama, err = self._one_rs(q)
            if err:
                return err
            h, rows = self.by_name[nama], self.by_rs.get(nama, [])
            if rx:
                rows = [r for r in rows if rx.search(r["text"])]
            a = self._agg(rows)
            hasil.append({"nama": nama, "alamat": h["alamat"] or "belum ada di data", "kecamatan": h["kec"],
                          "rating_semua_ulasan": round(h["rating"], 2) if h["rating"] is not None else None,
                          "jumlah_ulasan_semua": h["n"], "ulasan_dianalisis": a["n"],
                          "positif_pct": _pc(a["positif"], a["n"]), "negatif_pct": _pc(a["negatif"], a["n"]),
                          "keluhan_utama": self._keluhan(nama, 3), "sampel_kecil": a["n"] < 20})
        return {"topik": topik or "(semua ulasan)", "perbandingan": hasil}

    def topik_keluhan(self, nama_rs="", jumlah=5):
        nama, err = self._one_rs(nama_rs)
        if err:
            return err
        rows = self.by_rs.get(nama, [])
        neg = sum(r["sent"] == "negatif" for r in rows)
        out = {"rs": nama, "ulasan_teks": len(rows), "ulasan_negatif": neg,
               "topik": self._keluhan(nama, _cl(jumlah, 1, 8, 5))}
        if neg < 5:
            out["catatan"] = "sampel kecil (kurang dari 5 ulasan negatif)"
        return out

    def tren_rs(self, nama_rs="", bulan=6):
        nama, err = self._one_rs(nama_rs)
        if err:
            return err
        per = defaultdict(list)
        for r in self.by_rs.get(nama, []):
            if r.get("date") and r["sent"]:
                per[r["date"][:7]].append(r)
        if not per:
            return {"error": "ulasan RS ini belum punya tanggal"}
        bl = sorted(per)[-_cl(bulan, 2, 24, 6):]
        seri = []
        for b in bl:
            a = self._agg(per[b])
            seri.append({"bulan": b, "ulasan": a["n"], "rating": a["rating"],
                         "positif_pct": _pc(a["positif"], a["n"]), "negatif_pct": _pc(a["negatif"], a["n"])})
        half = len(bl) // 2
        awal = self._agg([r for b in bl[:half] for r in per[b]])
        akhir = self._agg([r for b in bl[half:] for r in per[b]])
        if half == 0 or awal["n"] < 10 or akhir["n"] < 10:
            kes = "data belum cukup untuk menyimpulkan tren (butuh minimal 10 ulasan di tiap paruh periode)"
        else:
            dp = _pc(akhir["positif"], akhir["n"]) - _pc(awal["positif"], awal["n"])
            dr = (akhir["rating"] or 0) - (awal["rating"] or 0)
            kes = "membaik" if (dp >= 5 or dr >= 0.2) else "memburuk" if (dp <= -5 or dr <= -0.2) else "relatif stabil"
        return {"rs": nama, "bulan_dengan_data": seri, "kesimpulan": kes}

    def info_rs(self, nama_rs=""):
        nama, err = self._one_rs(nama_rs)
        if err:
            return err
        i = self.info.get(nama)
        if not i:
            k = _key(nama)
            i = next((v for n, v in self.info.items() if _key(n) == k), None)
        jam = i.get("hours") or []
        h24 = bool(jam) and len(jam) >= 7 and all(
            ("24 jam" in str(x).lower() or "24 hours" in str(x).lower()) for x in jam)
        h = self.by_name[nama]
        return {"nama": nama, "alamat": h["alamat"] or "belum ada di data", "kecamatan": h["kec"],
                "rating": round(h["rating"], 2) if h["rating"] is not None else None,
                "jumlah_ulasan": h["n"],
                "telepon": i.get("phone"), "website": i.get("website"),
                "kategori": i.get("categories"), "jam_buka": jam[:7],
                "buka_24_jam": h24 if jam else None, "diperbarui": i.get("updated"), "sumber": "Google Maps"}

    def _find_rs(self, q):
        res = super()._find_rs(q)
        if len(res) == 1:
            return res
        q2 = re.sub(r"\brsu\b", "rumah sakit umum", (q or "").lower())
        if q2 != (q or "").lower():
            r2 = super()._find_rs(q2)
            if r2:
                return r2
        return res

    # ---- dispatch + catatan panggilan (untuk log pertanyaan)
    def call(self, name, args, user_latlng=None):
        args = args if isinstance(args, dict) else {}
        fns = {"rekomendasi": lambda: self.rekomendasi(user_latlng=user_latlng, **args),
               "bandingkan_rs": lambda: self.bandingkan_rs(**args),
               "topik_keluhan": lambda: self.topik_keluhan(**args),
               "tren_rs": lambda: self.tren_rs(**args),
               "info_rs": lambda: self.info_rs(**args)}
        if name in fns:
            try:
                result = fns[name]()
            except TypeError as e:
                result = {"error": f"argumen tidak valid: {e}"}
            except Exception as e:
                result = {"error": f"alat gagal: {e}"}
        else:
            result = super().call(name, args, user_latlng)
        calls = CALLS.get()
        if calls is not None:
            empty = isinstance(result, dict) and (
                "error" in result or not any(v for k, v in result.items() if k != "catatan"))
            calls.append((name, empty))
        return result