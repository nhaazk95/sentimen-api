"""
cek_model_aktif.py
Menampilkan model apa saja yang ada di database (Neon) dan mana yang aktif.
"""
import os
from sqlalchemy import text
from ops_agent import make_engine

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL belum diset. Set dulu di .env atau environment.")
    
    print("[info] Menghubungkan ke database...")
    engine = make_engine()
    
    query = """
        SELECT version, created_at, active, metrics 
        FROM models 
        ORDER BY created_at DESC
    """
    
    with engine.connect() as c:
        rows = c.execute(text(query)).all()
        
    if not rows:
        print("⚠️ Tidak ada model di tabel 'models'.")
        print("Artinya, API saat ini menggunakan model dari file .pkl (svm_pipeline_final.pkl).")
        return
        
    print(f"\nDitemukan {len(rows)} model di database:")
    print("-" * 80)
    for r in rows:
        status = "AKTIF" if r.active == 1 else "nonaktif"
        print(f"Versi: {r.version} | Tanggal: {r.created_at} | Status: {status}")
        print(f"Metrics: {r.metrics}")
        print("-" * 80)

if __name__ == "__main__":
    main()