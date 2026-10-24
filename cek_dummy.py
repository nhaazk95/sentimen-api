import pandas as pd
from sqlalchemy import text
from ops_agent import make_engine
e = make_engine()
q = {
 "Dummy di reviews": "SELECT review_id, notified, scraped_at FROM reviews WHERE review_id LIKE 'TEST-%'",
 "Dummy di notif_log": "SELECT event, review_id, ts FROM notif_log WHERE review_id LIKE 'TEST-%' ORDER BY ts",
 "Antrean": "SELECT COUNT(*) AS menunggu FROM reviews WHERE LOWER(COALESCE(corrected_label, label)) = 'negatif' AND notified = 0",
}
for n, s in q.items():
    print("\n==", n, "=="); print(pd.read_sql(text(s), e).to_string(index=False))