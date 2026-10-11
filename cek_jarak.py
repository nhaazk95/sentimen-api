# python cek_jarak.py LAT LON   (titik tes: klik kanan di Google Maps > salin koordinat)
import sys
import main as m

print("== RS dengan koordinat kosong atau di luar kisaran kasar Banyumas ==")
for h in m.RS_HOSPITALS:
    if h["lat"] is None or not (-8.0 < h["lat"] < -7.0 and 108.7 < h["lon"] < 109.7):
        print(" ", h["nama"], h["lat"], h["lon"])

lat, lon = float(sys.argv[1]), float(sys.argv[2])
rs = sorted(((m._haversine(lat, lon, h["lat"], h["lon"]), h) for h in m.RS_HOSPITALS if h["lat"] is not None),
            key=lambda x: x[0])
print(f"\n== 8 RS terdekat dari ({lat}, {lon}) ==")
for d, h in rs[:8]:
    print(f'{d:5.1f} km  {h["nama"]}  | kec={h["kec"]!r}  | ({h["lat"]:.5f}, {h["lon"]:.5f})')