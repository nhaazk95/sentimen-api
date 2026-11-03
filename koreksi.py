import pandas as pd
import hashlib
from sqlalchemy import update, select, create_engine
from ops_agent import reviews as REVIEWS_TBL

# --- TANAM URL NEON LANGSUNG ---
DATABASE_URL = "postgresql+psycopg2://neondb_owner:npg_dGcIY29PefSi@ep-quiet-glitter-b5hee6zi-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  
# -------------------------------

print(f"Menggunakan database: {DATABASE_URL[:50]}...")

engine = create_engine(DATABASE_URL)

df = pd.read_excel("kandidat_koreksi.xlsx")
print(f"Total baris Excel: {len(df)}")

berhasil_update = 0
berhasil_insert = 0
gagal = 0

print("Memproses data...")

with engine.begin() as conn:
    for index, row in df.iterrows():
        teks = str(row['teks']).strip()
        label_baru = str(row['label_baru']).strip()
        rs_name = str(row['rs']).strip()
        rating = row['rating']
        tgl = str(row['tgl'])
        
        if not teks or teks == 'nan' or not label_baru:
            continue

        kata_kunci = " ".join(teks.split()[:5])
        query = select(REVIEWS_TBL.c.review_id).where(REVIEWS_TBL.c.text.ilike(f"%{kata_kunci}%"))
        result = conn.execute(query).fetchone()
        
        if result:
            review_id = result[0]
            stmt = (
                update(REVIEWS_TBL)
                .where(REVIEWS_TBL.c.review_id == review_id)
                .values(corrected_label=label_baru, label_source='manusia')
            )
            conn.execute(stmt)
            berhasil_update += 1
        else:
            new_id = hashlib.sha256(f"{rs_name}|{teks}".encode()).hexdigest()[:16]
            stmt = REVIEWS_TBL.insert().values(
                review_id=new_id,
                rs_name=rs_name,
                rating=rating,
                text=teks,
                published_at=tgl,
                label='Negatif',
                corrected_label=label_baru,
                label_source='manusia',
                notified=1,
                legacy=1
            )
            try:
                conn.execute(stmt)
                berhasil_insert += 1
            except Exception as e:
                gagal += 1
                print(f"[GAGAL INSERT] {new_id}: {e}")

print("---------------------------------")
print(f"Berhasil UPDATE: {berhasil_update}")
print(f"Berhasil INSERT: {berhasil_insert}")
print(f"Gagal: {gagal}")