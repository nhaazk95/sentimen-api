import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, update, select
from ops_agent import reviews as REVIEWS_TBL
from preprocessing import full_preprocess, load_slang_dict

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
slang_dict = load_slang_dict('slang_dict.json')

def preprocess(text):
    return full_preprocess(text, slang_dict)

print("Mengisi kolom clean untuk koreksi manual...")

with engine.begin() as conn:
    # Ambil semua koreksi manual yang clean-nya kosong
    rows = conn.execute(select(REVIEWS_TBL.c.review_id, REVIEWS_TBL.c.text)
                        .where(REVIEWS_TBL.c.corrected_label.is_not(None),
                               REVIEWS_TBL.c.clean.is_(None))).all()
    
    print(f"Ditemukan {len(rows)} ulasan dengan clean kosong.")
    
    for row in rows:
        review_id = row[0]
        text = row[1]
        clean = preprocess(text)
        
        stmt = update(REVIEWS_TBL).where(REVIEWS_TBL.c.review_id == review_id).values(clean=clean)
        conn.execute(stmt)

print("Selesai!")