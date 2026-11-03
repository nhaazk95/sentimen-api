"""
matikan_model_db.py
Menonaktifkan model lama di database (set active = 0) agar API
kembali menggunakan model dari file .pkl (svm_pipeline_final.pkl).

TIDAK menghapus data ulasan, run, atau tabel lain.
Hanya menonaktifkan model di tabel `models`.
"""

import os
import sys
from sqlalchemy import select, update

# Pastikan folder proyek ada di path agar `ops_agent` bisa di-import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ops_agent import make_engine, models
except ImportError as e:
    print(f"[ERROR] Gagal import ops_agent: {e}")
    print("Pastikan file ini dijalankan di folder yang sama dengan ops_agent.py")
    sys.exit(1)

def main():
    # 1. Hubungkan ke database
    print("[info] Menghubungkan ke database...")
    try:
        engine = make_engine()
        # Tes koneksi
        with engine.connect() as c:
            c.execute(select(1))
        print("[ok] Koneksi berhasil.")
    except Exception as e:
        print(f"[ERROR] Gagal koneksi ke database: {e}")
        print("Pastikan DATABASE_URL sudah diset (atau pakai sqlite lokal).")
        sys.exit(1)

    # 2. Cek model yang aktif saat ini
    print("\n[info] Status model SEBELUM diubah:")
    with engine.connect() as c:
        rows = c.execute(
            select(models.c.version, models.c.created_at, models.c.active)
        ).all()
    
    if not rows:
        print("  (Tidak ada model di database. API sudah otomatis pakai .pkl)")
    else:
        for r in rows:
            status = "AKTIF" if r.active == 1 else "nonaktif"
            print(f"  - Versi: {r.version} | Tanggal: {r.created_at} | Status: {status}")

    # 3. Nonaktifkan semua model
    print("\n[info] Menonaktifkan semua model di database...")
    with engine.begin() as c:
        hasil = c.execute(update(models).values(active=0))
        print(f"[ok] {hasil.rowcount} baris diperbarui.")

    # 4. Verifikasi
    print("\n[info] Status model SESUDAH diubah:")
    with engine.connect() as c:
        rows = c.execute(
            select(models.c.version, models.c.created_at, models.c.active)
        ).all()
        for r in rows:
            status = "AKTIF" if r.active == 1 else "nonaktif"
            print(f"  - Versi: {r.version} | Tanggal: {r.created_at} | Status: {status}")

    print("\n" + "="*60)
    print("SELESAI.")
    print("="*60)
    print("Langkah selanjutnya:")
    print("  1. Restart server API Anda (Render/VPS/lokal).")
    print("  2. Cek log: TIDAK BOLEH ada tulisan '[ops] memakai model dari database'")
    print("  3. Tes endpoint /predict, pastikan 'f1_macro' = 0.7837")
    print("  4. Cek bahwa 'accuracy' = 0.9450 (bukan 0.9927)")
    print("\nData ulasan, riwayat run, dan tabel lain TIDAK terhapus.")

if __name__ == "__main__":
    main()