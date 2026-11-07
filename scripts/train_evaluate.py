# label_gabungan.py
"""
Pelabelan LLM untuk data_gabungan_final.csv.
Menggunakan kolom 'clean' (hasil preprocessing). Fallback ke 'teks'
kalau clean kosong.
"""
import os, json, time
import pandas as pd
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"
BATCH, OUT = 10, "label_llm.csv"
SYSTEM = open("prompt_label.txt", encoding="utf-8").read()

DATA_IN = "data_gabungan_final.csv"

# ---------- baca ----------
df = pd.read_csv(DATA_IN).reset_index().rename(columns={"index": "id"})
print(f"[INFO] {len(df)} baris")

# pilih teks untuk LLM: 'clean' jika ada & terisi, fallback 'teks'
if "clean" in df.columns:
    teks = df["clean"].where(df["clean"].notna() & (df["clean"].astype(str).str.strip() != ""), df["teks"])
    print(f"[INFO] pakai 'clean' ({df['clean'].notna().sum()} terisi), fallback 'teks'")
else:
    teks = df["teks"]
    print("[WARN] kolom 'clean' tidak ada, pakai 'teks' mentah")

df["teks_llm"] = teks.astype(str).str[:1500]

# hanya bertext
df = df[df["teks_llm"].str.strip() != ""]
print(f"[INFO] {len(df)} baris bertext")

# ---------- resume ----------
done = set(pd.read_csv(OUT)["id"]) if os.path.exists(OUT) else set()
todo = df[~df["id"].isin(done)]
print(f"[INFO] sudah: {len(done)} | perlu: {len(todo)}")

# ---------- prioritas Mei 2026 ----------
if "tanggal_ulasan" in todo.columns:
    todo["tanggal_ulasan"] = pd.to_datetime(todo["tanggal_ulasan"], errors="coerce")
    is_mei = (todo["tanggal_ulasan"].dt.year == 2026) & (todo["tanggal_ulasan"].dt.month == 5)
    todo = pd.concat([todo[is_mei], todo[~is_mei]]).reset_index(drop=True)
    print(f"[INFO] Mei 2026 diprioritaskan: {is_mei.sum()} baris")

# ---------- panggil ----------
def panggil(rows):
    payload = [{"id": int(r.id), "teks": r.teks_llm} for r in rows.itertuples()]
    for percobaan in range(6):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
            hasil = json.loads(r.choices[0].message.content)["hasil"]
            return [(h["id"], h["label"]) for h in hasil
                    if h["label"] in {"positif", "netral", "negatif"}]
        except Exception as e:
            wait = min(60, 5 * 2 ** percobaan)
            print(f"  [retry {percobaan+1}/6] {e} -> {wait}s")
            time.sleep(wait)
    return []

# ---------- loop ----------
for i in range(0, len(todo), BATCH):
    batch = todo.iloc[i:i + BATCH]
    res = panggil(batch)
    if res:
        pd.DataFrame(res, columns=["id", "label_llm"]).to_csv(
            OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(f"  batch {i//BATCH+1}: {len(res)}/{len(batch)}")
    time.sleep(2)

# ---------- cek gagal ----------
if os.path.exists(OUT):
    hasil = pd.read_csv(OUT)
    gagal = df[~df["id"].isin(hasil["id"])]
    print(f"\n[DONE] terlabel: {len(hasil)} | gagal: {len(gagal)}")
    if len(gagal):
        gagal[["id", "teks_llm"]].to_csv("perlu_ulang.csv", index=False)