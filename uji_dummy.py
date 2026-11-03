from datetime import datetime, timezone
from sqlalchemy import text
from ops_agent import make_engine   # samakan dengan impor di cek_metrik.py

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
e = make_engine()
with e.begin() as c:
    c.execute(text("""
        INSERT INTO reviews (review_id, rs_name, rating, text, clean, label,
                             confidence, published_at, scraped_at, notified, legacy)
        VALUES ('TEST-002', 'RSUD Ajibarang', 1,
                '[UJI SISTEM] Pelayanan lambat dan petugas kurang ramah.',
                'uji sistem pelayanan lambat petugas kurang ramah',
                'Negatif', 1.0, :t, :t, 0, 0)"""), {"t": now})

print("dummy dimasukkan:", now)