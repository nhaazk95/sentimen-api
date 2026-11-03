import pandas as pd
from sqlalchemy import text
from ops_agent import make_engine   # sesuaikan dengan nama file modulmu

engine = make_engine()             # membaca DATABASE_URL dari environment
print("DB:", engine.dialect.name)

queries = {
    "Keluhan diambil / ditandai terkirim": """
        SELECT event, COUNT(DISTINCT review_id) AS jumlah
        FROM notif_log GROUP BY event""",
    "Duplikat terkirim (harus kosong)": """
        SELECT review_id, COUNT(*) AS kali
        FROM notif_log WHERE event = 'marked'
        GROUP BY review_id HAVING COUNT(*) > 1""",
    "Diambil tapi tidak ditandai": """
        SELECT DISTINCT review_id FROM notif_log WHERE event = 'served'
        EXCEPT SELECT review_id FROM notif_log WHERE event = 'marked'""",
    "Status run di server": """
        SELECT status, COUNT(*) AS jumlah FROM agent_runs GROUP BY status""",
    "Rincian run tidak sukses": """
        SELECT started_at, status, SUBSTR(report, 1, 300) AS penyebab
        FROM agent_runs WHERE status <> 'ok' ORDER BY started_at""",
    "Run per hari": """
        SELECT SUBSTR(started_at, 1, 10) AS tanggal, status, COUNT(*) AS jumlah
        FROM agent_runs GROUP BY 1, 2 ORDER BY 1""",
    "Durasi proses tahap 1": """
        SELECT started_at, status,
            ROUND(EXTRACT(EPOCH FROM (finished_at::timestamptz
                    - started_at::timestamptz))::numeric, 1) AS detik
        FROM agent_runs
        WHERE started_at >= '2026-10-17' ORDER BY started_at""",
    "Ketepatan waktu (scraped_at -> ditandai)": """
        SELECT r.rs_name, r.scraped_at, n.ts AS ditandai,
            ROUND(EXTRACT(EPOCH FROM (n.ts::timestamptz
                    - r.scraped_at::timestamptz))/60) AS menit
        FROM reviews r JOIN notif_log n
        ON n.review_id = r.review_id AND n.event = 'marked'
        ORDER BY n.ts""",
    "Ukuran data": """
        SELECT legacy, COUNT(*) AS jumlah FROM reviews GROUP BY legacy""",
    "Log langkah per run": """
        SELECT started_at, SUBSTR(log, 1, 700) AS log FROM agent_runs
        WHERE started_at BETWEEN '2026-10-20T22:00:00Z' AND '2026-10-21T12:00:00Z'
        ORDER BY started_at""",
    "Riwayat retrain": """
            SELECT ts, durasi_detik, dipasang, versi FROM retrain_log ORDER BY ts""",
        "Run dengan Apify gagal": """
        SELECT started_at, status FROM agent_runs
        WHERE log LIKE '%Apify gagal%' ORDER BY started_at""",
    "Ulasan baru dan negatif per run": """
        SELECT started_at,
            (regexp_match(log, '''baru'': ([0-9]+)'))[1] AS baru,
            (regexp_match(log, '''Negatif'': ([0-9]+)'))[1] AS negatif,
            (log LIKE '%Apify gagal%') AS apify_gagal
        FROM agent_runs WHERE started_at >= '2026-10-11' ORDER BY started_at""",
    "Negatif belum diemail": """
        SELECT COUNT(*) AS menunggu, MIN(scraped_at) AS tertua FROM reviews
        WHERE LOWER(COALESCE(corrected_label, label)) = 'negatif' AND notified = 0""",
    "Model aktif": """
        SELECT version, created_at, active, metrics FROM models ORDER BY created_at""",
    "Aktivitas notif_log": """
        SELECT event, COUNT(*) AS baris, MAX(ts) AS terakhir
        FROM notif_log GROUP BY event""",
    "Cek dummy": """SELECT review_id, notified FROM reviews WHERE review_id LIKE 'TEST-%'""",
    "Teks asli ulasan uji": """
        SELECT rs_name, rating, published_at, text FROM reviews
        WHERE scraped_at >= '2026-10-23T14:00:00Z'
        AND LOWER(COALESCE(corrected_label, label)) = 'negatif'
        ORDER BY scraped_at""",
}

for nama, q in queries.items():
    print(f"\n== {nama} ==")
    print(pd.read_sql(text(q), engine).to_string(index=False))