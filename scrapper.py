# scrape_reviews.py
"""
Scrape ulasan Google Maps via Apify.
Output: ulasan.csv dengan kolom: rs, user, rating, tanggal_ulasan, teks

Cara pakai:
    $env:APIFY_TOKEN="apify_api_..."
    python scrape_reviews.py
"""
import os, json, time
from urllib.parse import urlparse, parse_qs, unquote
import pandas as pd
from apify_client import ApifyClient

# ============================================================
# KONFIGURASI
# ============================================================
TOKEN = os.environ["APIFY_TOKEN"]
client = ApifyClient(TOKEN)

RS_INPUT    = "places.json"
ULASAN_CSV  = "ulasan.csv"
GAGAL_CSV   = "rs_gagal.csv"

MAX_REVIEWS = 100
LANGUAGE    = "id"
SORT        = "newest"
TANGGAL_MIN = "2026-05-01"
SLEEP_ANTAR = 3

REVIEWS_ACTOR = "compass/google-maps-reviews-scraper"   # ganti kalau nama beda

# ============================================================
# HELPERS
# ============================================================
def query_from_url(url: str) -> str:
    q = parse_qs(urlparse(url).query).get("query", [""])[0]
    return unquote(q).replace("+", " ")

# ============================================================
# SCRAPE
# ============================================================
def scrape_one(url: str, nama: str) -> list[dict]:
    run = client.actor(REVIEWS_ACTOR).call(run_input={
        "startUrls": [{"url": url}],
        "maxReviews": MAX_REVIEWS,
        "language": LANGUAGE,
        "reviewsSort": SORT,
        "reviewsStartDate": TANGGAL_MIN,
    })
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    rows = []
    for it in items:
        teks = (it.get("text") or "").strip()
        if not teks:
            continue
        rows.append({
            "rs": nama,
            "user": it.get("name") or it.get("reviewerName") or "",
            "rating": it.get("stars"),
            "tanggal_ulasan": it.get("publishedAtDate") or it.get("publishAt"),
            "teks": teks,
        })
    print(f"  {nama}: {len(rows)}/{len(items)} bertext")
    return rows

def main():
    with open(RS_INPUT, encoding="utf-8") as f:
        items = json.load(f)

    print(f"[INFO] {len(items)} RS, max {MAX_REVIEWS}/RS, actor={REVIEWS_ACTOR}\n")

    all_rows, gagal = [], []
    for i, it in enumerate(items, 1):
        url = it["url"]
        nama = query_from_url(url) or f"RS_{i}"
        try:
            rows = scrape_one(url, nama)
            if not rows:
                gagal.append({"nama": nama, "url": url, "alasan": "0 ulasan bertext"})
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [ERROR] {nama}: {e}")
            gagal.append({"nama": nama, "url": url, "alasan": str(e)[:200]})
        time.sleep(SLEEP_ANTAR)

    if not all_rows:
        print("\n[WARN] tidak ada data terambil")
        if gagal:
            pd.DataFrame(gagal).to_csv(GAGAL_CSV, index=False)
            print(f"[INFO] detail gagal -> {GAGAL_CSV}")
        return

    df = pd.DataFrame(all_rows)
    df["tanggal_ulasan"] = pd.to_datetime(df["tanggal_ulasan"], errors="coerce", utc=True)

    n0 = len(df)
    df = df[df["tanggal_ulasan"] >= TANGGAL_MIN]
    print(f"\n[FILTER] >= {TANGGAL_MIN}: {len(df)}/{n0}")

    df = df[["rs", "user", "rating", "tanggal_ulasan", "teks"]].sort_values("tanggal_ulasan")
    df["tanggal_ulasan"] = df["tanggal_ulasan"].dt.strftime("%Y-%m-%d")

    # Append + dedup kalau file sudah ada (untuk scraping ulang)
    if os.path.exists(ULASAN_CSV):
        lama = pd.read_csv(ULASAN_CSV)
        df = pd.concat([lama, df], ignore_index=True)
        df = df.drop_duplicates(subset=["rs", "teks"])

    df.to_csv(ULASAN_CSV, index=False)

    print(f"\n[DONE] total {len(df)} -> {ULASAN_CSV}")
    print("\nPer RS:")
    print(df.groupby("rs").size().sort_values(ascending=False).to_string())

    if gagal:
        pd.DataFrame(gagal).to_csv(GAGAL_CSV, index=False)
        print(f"\n[INFO] {len(gagal)} RS gagal -> {GAGAL_CSV}")

if __name__ == "__main__":
    main()