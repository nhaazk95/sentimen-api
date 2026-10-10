"""
prepare_training_data.py -- bersihkan train_pool.csv dan test_set.csv SEKALI di laptop, supaya retrain di
server tidak perlu menjalankan stemming ribuan kalimat (lambat di Render gratis).

Pakai (dari root repo sentimen-api):  python prepare_training_data.py
Hasil: data/training/train_pool_clean.csv dan test_set_clean.csv (kolom clean_text, label).
Retrain otomatis memakai file *_clean.csv bila ada. Jalankan ulang bila preprocessing.py atau slang_dict.json berubah.
"""
import pandas as pd

from preprocessing import full_preprocess, load_slang_dict

slang = load_slang_dict("slang_dict.json")
for name in ("train_pool", "test_set"):
    src = f"data/training/{name}.csv"
    df = pd.read_csv(src)
    out = pd.DataFrame({"clean_text": df["text"].astype(str).map(lambda t: full_preprocess(t, slang)),
                        "label": df["label"]}).dropna()
    out = out[out["clean_text"].str.strip().ne("")]
    out.to_csv(f"data/training/{name}_clean.csv", index=False)
    print(f"{src}: {len(df)} baris -> {len(out)} baris bersih")
