import pandas as pd
import uuid
import hashlib
from sqlalchemy import select, insert, update
from ops_agent import make_engine, reviews as REVIEWS_TBL

ENGINE = make_engine()

print("Membaca file Excel...")
df = pd.read_excel("kandidat_koreksi.xlsx")

# Ambil hanya kolom yang dibutuhkan
# Pastikan nama kolom sesuai dengan file Excel-mu
df = df.rename(columns={
    df.columns[0]: 'id_lama', 
    'rs': 'rs_name', 
    'rating': 'rating', 
    'tgl': 'published_at', 
    'teks': 'text', 
    'label_baru': 'corrected_label'
})

berhasil_update = 0
berhasil_insert = 0
gagal = 0

print("Memproses data...")

with ENGINE.begin() as conn:
    for index, row in df.iterrows():
        teks = str(row['text']).strip()
        label_baru = str(row['corrected_label']).strip()
        rs_name = str(row['rs_name']).strip()
        rating = row['rating']
        tgl = row['published_at']
        
        if not teks or teks == 'nan' or not label_baru:
            continue

        # 1. Cek apakah ulasan sudah ada di database (berdasarkan teks persis)
        query = select(REVIEWS_TBL.c.review_id).where(REVIEWS_TBL.c.text == teks)
        result = conn.execute(query).fetchone()
        
        if result:
            # Jika sudah ada, update corrected_label
            review_id = result[0]
            stmt = (
                update(REVIEWS_TBL)
                .where(REVIEWS_TBL.c.review_id == review_id)
                .values(corrected_label=label_baru, label_source='manusia')
            )
            conn.execute(stmt)
            berhasil_update += 1
        else:
            # Jika belum ada, masukkan sebagai data baru
            # Buat ID unik dari hash teks (karena ID asli tidak diketahui)
            new_id = hashlib.sha256(f"{rs_name}|{teks}".encode()).hexdigest()[:16]
            
            stmt = insert(REVIEWS_TBL).values(
                review_id=new_id,
                rs_name=rs_name,
                rating=rating,
                text=teks,
                published_at=str(tgl),
                label='Negatif',           # Label prediksi awal (karena ini kandidat negatif)
                corrected_label=label_baru, # Label koreksi manusia
                label_source='manusia',
                notified=1,                 # Anggap sudah dikirim (karena ini data lama)
                legacy=1                    # Tandai sebagai data lama
            )
            conn.execute(stmt)
            berhasil_insert += 1

print("---------------------------------")
print(f"Selesai!")
print(f"Berhasil UPDATE (sudah ada): {berhasil_update} ulasan.")
print(f"Berhasil INSERT (data baru): {berhasil_insert} ulasan.")
print(f"Total diproses: {berhasil_update + berhasil_insert} ulasan.")