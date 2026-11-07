import pandas as pd

# 1. Load dataset
file_name = "train_fixed.csv"  # Ganti dengan nama file kamu
df = pd.read_csv(file_name)

# 2. Tentukan kata yang mau diubah
kata_lama = "memuaskan"
kata_baru = "muas"

# 3. Ganti di kolom 'clean' dan 'teks'
# (Gunakan regex=False agar aman jika ada karakter khusus)
for kolom in ['clean', 'teks']:
    if kolom in df.columns:
        df[kolom] = df[kolom].astype(str).str.replace(kata_lama, kata_baru, regex=False)

# 4. Simpan ke file baru (atau timpa file lama)
df.to_csv("train_fixed.csv", index=False, encoding="utf-8-sig")
print(f"Berhasil mengubah '{kata_lama}' menjadi '{kata_baru}'")