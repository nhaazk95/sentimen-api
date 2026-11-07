import os
from dotenv import load_dotenv
load_dotenv()
from ops_agent import make_engine, reviews, OpsTools

engine = make_engine()
print("URL DB:", engine.url)

# Cek total negatif belum terkirim langsung dari SQL
with engine.connect() as c:
    from sqlalchemy import select
    rows = c.execute(select(reviews.c.review_id, reviews.c.notified).where(reviews.c.notified == 0)).all()
    print("Ulasan notified=0 di DB:", len(rows))
    for r in rows:
        print(r)