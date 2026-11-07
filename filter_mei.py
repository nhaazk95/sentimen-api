# update_tanggal_mei.py
"""
Update tanggal_ulasan di dataset final berdasarkan ulasan_mei2026.csv.
Cocokkan berdasarkan (rs, teks) persis.
"""
import pandas as pd

FINAL = "data_gabungan_final.csv"
MEI   = "ulasan_mei2026.csv"

def main():
    final = pd.read_csv(FINAL)
    mei   = pd.read_csv(MEI)

    print(f"[INFO] final: {len(final)} baris | mei: {len(mei)} baris")

    # Map (rs, teks) -> tanggal dari Mei
    mei_map = {(str(r.rs).strip(), str(r.teks)): r.tanggal_ulasan for r in mei.itertuples()}

    # Update tanggal di final kalau ada di map
    n_updated = 0
    n_missing = 0
    for i, row in final.iterrows():
        key = (str(row["rs"]).strip(), str(row["teks"]))
        if key in mei_map:
            if str(row["tanggal_ulasan"]) != mei_map[key]:
                final.at[i, "tanggal_ulasan"] = mei_map[key]
                n_updated += 1
        else:
            n_missing += 1

    print(f"[UPDATE] {n_updated} baris tanggal diperbarui")
    print(f"[INFO] {n_missing} baris tidak ada pasangannya di Mei (normal)")

    final.to_csv(FINAL, index=False)
    print(f"[DONE] -> {FINAL}")

    # Verifikasi: hitung ulang per bulan
    final["tanggal_ulasan"] = pd.to_datetime(final["tanggal_ulasan"], errors="coerce")
    final["bulan"] = final["tanggal_ulasan"].dt.to_period("M")
    print("\nRingkasan per bulan (terakhir):")
    print(final.groupby("bulan").size().tail(8).to_string())

    # Cek khusus Mei 2026
    mei_count = (final["bulan"] == "2026-05").sum()
    print(f"\n[CEK] Baris bulan Mei 2026 di final: {mei_count}")

if __name__ == "__main__":
    main()