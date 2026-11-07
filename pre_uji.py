import pandas as pd
from preprocessing import full_preprocess, load_slang_dict

# 1. Load data uji mentah
test = pd.read_csv("test.csv")
print("Jumlah data uji:", len(test))

# 2. Preprocessing (karena di test2 belum ada kolom 'clean')
print("Melakukan preprocessing...")
slang_dict = load_slang_dict('slang_dict.json')
test['clean'] = test['text'].fillna("").apply(lambda x: full_preprocess(x, slang_dict))

# 4. Simpan
test.to_csv("test2_siap_label.csv", index=False)
print("-> File 'test2_siap_label.csv' siap diisi label manual!")