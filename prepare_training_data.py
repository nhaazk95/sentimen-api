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
