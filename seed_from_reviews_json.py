"""
seed_from_reviews_json.py -- impor SEKALI isi reviews.json lama ke database agen, supaya dashboard tidak
kosong dan ulasan lama tidak masuk dua kali saat Apify mengambilnya lagi.

Pakai (file lokal atau URL):
    DATABASE_URL=postgresql://... python seed_from_reviews_json.py reviews.json
    DATABASE_URL=postgresql://... python seed_from_reviews_json.py https://.../data/reviews.json

Koreksi manual lama (kolom Label_Manual) ikut diimpor sebagai label manusia, sehingga dipakai untuk retrain.
"""
import hashlib
import json
import sys
import urllib.request

from sqlalchemy import insert, select

from ops_agent import _now, init_db, make_engine, reviews


def load(src):
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    with open(src, encoding="utf-8") as f:
        return json.load(f)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main(src):
    data = load(src)
    if not isinstance(data, list) or not data:
        sys.exit("reviews.json kosong atau bukan berupa daftar.")
    rows, seen = [], set()
    for r in data:
        rs, txt = str(r.get("Nama RS") or "").strip(), str(r.get("Isi Ulasan") or "").strip()
        if not rs or not txt:
            continue
        rid = str(r.get("id") or "") or "seed-" + hashlib.sha1(f"{rs}|{txt}".encode("utf-8")).hexdigest()[:20]
        if rid in seen:
            continue
        seen.add(rid)
        rows.append({
            "review_id": rid, "rs_name": rs, "username": r.get("Username"), "rating": num(r.get("Rating")),
            "published_at": r.get("Waktu Ulasan"), "address": r.get("Lokasi Tempat"),
            "lat": num(r.get("Latitude")), "lon": num(r.get("Longitude")), "text": txt,
            "label": r.get("Sentimen_Prediksi"), "confidence": num(r.get("Confidence")),
            "corrected_label": r.get("Label_Manual") or None,
            "label_source": "manusia" if r.get("Label_Manual") else None,
            "processed_at": r.get("Processed_At"), "clean": r.get("Teks_Bersih"),
            "scraped_at": _now(),
            "notified": 0 if r.get("Notified") is False else 1,   # hormati status email yang belum terkirim
            "legacy": 1,
        })

    engine = make_engine()
    init_db(engine)
    with engine.begin() as c:
        have = set(c.execute(select(reviews.c.review_id)).scalars())
        new = [r for r in rows if r["review_id"] not in have]
        for i in range(0, len(new), 500):
            c.execute(insert(reviews), new[i:i + 500])
    print(f"Selesai: {len(new)} ulasan diimpor ({len(rows) - len(new)} sudah ada).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])