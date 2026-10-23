import pandas as pd, requests
rows = requests.get("https://sentimen-api.onrender.com/reviews.json", timeout=60).json()
df = pd.DataFrame(rows)[["id", "Nama RS", "Rating", "Waktu Ulasan", "Isi Ulasan"]]
df = df[df["Isi Ulasan"].astype(str).str.len() >= 25]
known = set()
for f in ("train_pool.csv", "test_set.csv"):
    known |= set(pd.read_csv(f"data/training/{f}")["text"].astype(str).str.strip().str.lower())
df = df[~df["Isi Ulasan"].astype(str).str.strip().str.lower().isin(known)]
df.sample(n=min(200, len(df)), random_state=42).assign(label_manual="").to_csv("sampel_label.csv", index=False)