# preprocess.py
"""
Preprocessing teks untuk data_gabungan_final.csv.
Output: kolom 'clean' (atau timpa kalau sudah ada).

Aturan: case folding, hapus <br>, URL, mention, hashtag, angka, tanda baca,
normalisasi spasi. Sesuaikan dengan pipeline yang Anda pakai sebelumnya
(untuk konsistensi dengan data lama).
"""
import re
import pandas as pd

DATA = "data_gabungan_final.csv"
KOLOM_IN = "teks"
KOLOM_OUT = "clean"

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)        # hapus <br>
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)        # hapus URL
    s = re.sub(r"@\w+|#\w+", " ", s)                    # hapus mention/hashtag
    s = re.sub(r"\d+", " ", s)                          # hapus angka
    s = re.sub(r"[^\w\s]", " ", s)                      # hapus tanda baca
    s = s.lower()                                        # case folding
    s = re.sub(r"\s+", " ", s).strip()                  # normalisasi spasi
    return s

def main():
    df = pd.read_csv(DATA)
    print(f"[INFO] {len(df)} baris")

    if KOLOM_OUT in df.columns:
        print(f"[INFO] kolom '{KOLOM_OUT}' sudah ada, akan ditimpa")

    df[KOLOM_OUT] = df[KOLOM_IN].apply(clean_text)

    # buang yang kosong setelah cleaning
    kosong = (df[KOLOM_OUT].str.strip() == "").sum()
    print(f"[INFO] {kosong} baris kosong setelah cleaning (tetap disimpan)")

    df.to_csv(DATA, index=False)
    print(f"[DONE] -> {DATA}")
    print(f"\nContoh:")
    print(df[[KOLOM_IN, KOLOM_OUT]].head(5).to_string())

if __name__ == "__main__":
    main()