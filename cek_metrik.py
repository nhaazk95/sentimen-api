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
}

for nama, q in queries.items():
    print(f"\n== {nama} ==")
    print(pd.read_sql(text(q), engine).to_string(index=False))