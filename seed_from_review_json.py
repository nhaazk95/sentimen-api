"""
seed_from_reviews_json.py -- impor SEKALI isi reviews.json lama ke database agen, supaya dashboard tidak
kosong dan ulasan lama tidak masuk dua kali saat Apify mengambilnya lagi.

Pakai (file lokal atau URL):
    DATABASE_URL=postgresql://... python seed_from_reviews_json.py reviews.json
    DATABASE_URL=postgresql://... python seed_from_reviews_json.py https://.../data/reviews.json

Juga membuatkan kerangka places.json (nama RS, alamat, koordinat) dari data yang sama.
Kamu tinggal mengisi "place_id" tiap RS.
"""
import hashlib
import json
import os
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
            "processed_at": r.get("Processed_At"), "clean": r.get("Teks_Bersih"),
            "scraped_at": _now(),
            "notified": 1,    # ulasan lama dianggap sudah diketahui: jangan diemailkan massal
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

    # kerangka places.json
    places = {}
    for r in rows:
        places.setdefault(r["rs_name"], {"name": r["rs_name"], "place_id": "", "address": r["address"],
                                         "lat": r["lat"], "lon": r["lon"]})
    path = os.environ.get("PLACES_FILE", "places.json")
    if os.path.exists(path):
        print(f"{path} sudah ada, tidak ditimpa.")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(places.values()), f, ensure_ascii=False, indent=2)
        print(f"{path} dibuat untuk {len(places)} RS. Isi kolom place_id tiap RS.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])