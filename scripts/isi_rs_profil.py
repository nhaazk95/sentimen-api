import json, os, re
from collections import defaultdict

REVIEWS = "../dashboard_rs_banyumas/docs/data/reviews.json"  # sesuaikan path
OUT = "data/rs_profil.json"

reviews = json.load(open(REVIEWS, encoding="utf-8"))
lama = {}
if os.path.exists(OUT):  # pertahankan isian manual (telepon, dll.)
    lama = {p["nama"]: p for p in json.load(open(OUT, encoding="utf-8"))}

grup = defaultdict(list)
for r in reviews:
    grup[r["Nama RS"]].append(r)

hasil = []
for nama, rows in sorted(grup.items()):
    bintang = [r["Rating"] for r in rows if r.get("Rating")]
    ref = next((r for r in rows if r.get("Latitude")), rows[0])
    alamat = ref.get("Lokasi Tempat")
    kec = re.search(r"Kec\.\s*([^,]+)", alamat or "")
    p = lama.get(nama, {})
    hasil.append({
        "nama": nama,
        "alamat": alamat,
        "kecamatan": kec.group(1).strip() if kec else None,
        "lat": ref.get("Latitude"),
        "lng": ref.get("Longitude"),
        "rating": round(sum(bintang) / len(bintang), 2) if bintang else None,
        "jumlah_ulasan": len(rows),
        "telepon": p.get("telepon"),
        "website": p.get("website"),
        "jam_buka": p.get("jam_buka"),
    })

json.dump(hasil, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(len(hasil), "RS ditulis ke", OUT)