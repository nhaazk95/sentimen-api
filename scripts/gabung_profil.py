import json, difflib

PROFIL = "data/rs_profil.json"
APIFY = "data/apify_profil.json"

profil = json.load(open(PROFIL, encoding="utf-8"))
apify = json.load(open(APIFY, encoding="utf-8"))
judul = {a["title"]: a for a in apify if a.get("title")}

def jam_ringkas(oh):
    if not oh:
        return None
    return "; ".join(f'{h["day"]} {h["hours"]}' for h in oh)

for p in profil:
    m = difflib.get_close_matches(p["nama"], judul.keys(), n=1, cutoff=0.5)
    if not m:
        print("TIDAK COCOK:", p["nama"])
        continue
    a = judul[m[0]]
    p["telepon"] = a.get("phone") or p["telepon"]
    p["website"] = a.get("website") or p["website"]
    p["jam_buka"] = jam_ringkas(a.get("openingHours")) or p["jam_buka"]
    print("OK:", p["nama"], "<-", m[0])

json.dump(profil, open(PROFIL, "w", encoding="utf-8"), ensure_ascii=False, indent=2)