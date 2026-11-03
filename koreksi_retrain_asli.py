import os
import json
import joblib
from sqlalchemy import create_engine, update, select
from ops_agent import OpsTools, init_db, kv
from preprocessing import full_preprocess, load_slang_dict

# --- TANAM URL NEON LANGSUNG ---
DATABASE_URL = "postgresql+psycopg2://neondb_owner:npg_dGcIY29PefSi@ep-quiet-glitter-b5hee6zi-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # <--- GANTI DENGAN URL DARI .env
# -------------------------------

print(f"Menggunakan database: {DATABASE_URL[:50]}...")

engine = create_engine(DATABASE_URL)
init_db(engine)

# Load model & preprocessing
pipeline = joblib.load('svm_pipeline.pkl')
label_encoder = joblib.load('label_encoder.pkl')
slang_dict = load_slang_dict('slang_dict.json')

def preprocess(text):
    return full_preprocess(text, slang_dict)

def get_model():
    return pipeline, label_encoder, {}

def deploy(pipe, le, metrics):
    global pipeline, label_encoder
    pipeline, label_encoder = pipe, le
    print("[deploy] Model baru dipasang di memori.")

# Buat OpsTools
OPS = OpsTools(engine, preprocess, get_model, deploy)

# --- PAKSA RESET last_retrain_count ---
with engine.begin() as conn:
    conn.execute(update(kv).where(kv.c.k == 'last_retrain_count').values(v='0'))
    # Kalau baris belum ada, insert
    result = conn.execute(select(kv.c.v).where(kv.c.k == 'last_retrain_count')).fetchone()
    if not result:
        conn.execute(kv.insert().values(k='last_retrain_count', v='0'))

print("Memulai retraining...")
hasil = OPS.retrain_model()
print("Hasil:", json.dumps(hasil, indent=2))