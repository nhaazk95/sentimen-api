import pandas as pd
import re

file_name = "test.csv" # Ganti dengan nama filemu

# 1. Baca paksa (skip baris yang rusak) dengan Python engine
try:
    df = pd.read_csv(file_name, engine='python', on_bad_lines='skip')
except Exception as e:
    print("Gagal:", e)
    exit()

# 2. Bersihkan karakter yang aneh (misal enter/newline di dalam teks)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))

# 3. Simpan ulang dengan format yang 100% benar
df.to_csv(file_name, index=False, encoding='utf-8-sig', quoting=1) 
# quoting=1 artinya setiap kolom teks akan diapit tanda kutip ("), jadi koma di dalam teks aman.
print("File berhasil diperbaiki dan dibersihkan!")