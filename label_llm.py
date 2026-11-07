# label_semua.py
"""
Pelabelan LLM untuk SEMUA baris di data_gabungan_final.csv.
- Prioritas Mei 2026 dulu, lalu sisanya (semua tetap dilabeli)
- Langsung tulis ke kolom label_llm + label_final di file yang sama
- Checkpoint hash-based (aman resume kalau Groq error/quota habis)
"""
import os, json, time, hashlib
import pandas as pd
from groq import Groq

# ============================================================
# KONFIGURASI
# ============================================================
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL       = "openai/gpt-oss-120b"
TEMPERATURE = 0.0
BATCH       = 10
MAX_CHARS   = 1500
SLEEP       = 2

DATA        = "train.csv"
CHECKPOINT  = "checkpoint_label.csv"
KOLOM_TEKS  = "clean"
KOLOM_LABEL = "label_llm"
KOLOM_FINAL = "label_final"

SYSTEM_PROMPT = """Anda adalah pelabel sentimen ulasan rumah sakit berbahasa Indonesia.

TUGAS
Beri SATU label sentimen untuk setiap ulasan, HANYA berdasarkan ISI TEKS.
Abaikan rating bintang, nama RS, dan nama pengulas.

KATEGORI
- "negatif": Teks memuat keluhan, kritik, kekecewaan, atau ketidakpuasan
  terhadap layanan. Contoh topik: waktu tunggu lama, sikap staf tidak ramah,
  biaya mahal/tidak transparan, fasilitas rusak/kotor, salah penanganan,
  janji tidak ditepati, sarkasme bernada negatif.
- "positif": Teks memuat kepuasan atau pujian, TANPA keluhan sama sekali.
  Contoh: pujian keramahan, kecepatan, kebersihan, hasil perawatan, pelayanan.
- "netral": Teks hanya berisi informasi, pertanyaan, ucapan terima kasih
  tanpa penilaian, teks kosong/tidak jelas, hanya emoji.

ATURAN WAJIB (JANGAN DILANGGAR)
1. BACA SELURUH KALIMAT, bukan hanya kata kunci. Jangan memberi label
   "positif" hanya karena ada kata "ramah" atau "bagus", jika konteks
   kalimat justru menyindir atau mengeluh.
2. Kenali SARKASME dan MAJAS. Contoh: "Mantap nunggunya sampai 4 jam",
   "Pelayanan cepat sekali (sampai lupa dipanggil)" → NEGATIF.
3. ATURAN DOMINAN KELUHAN: Jika dalam satu ulasan terdapat kalimat pujian 
   DAN kalimat keluhan (campuran), maka label WAJIB "negatif". Satu keluhan 
   sekecil apapun harus ditangkap sebagai bahan evaluasi pihak Rumah Sakit, 
   meskipun ulasan tersebut didominasi oleh pujian.
4. Jika teks kosong, hanya angka, hanya emoji, atau tidak jelas maknanya
   → NETRAL.
5. JANGAN menambah fakta, alasan, atau tafsiran yang tidak tertulis
   di teks. Jangan mengarang.
6. JANGAN mengubah teks asli. Hanya beri label.

FORMAT OUTPUT
Balas HANYA dengan JSON valid, tanpa teks pembuka atau penutup,
tanpa penjelasan, tanpa markdown code block. Format:
{"hasil":[{"id":<id>,"label":"positif|netral|negatif"}]}
"""

VALID = {"positif", "netral", "negatif"}
MAP_KAPITAL = {"positif": "Positif", "netral": "Netral", "negatif": "Negatif"}

# ============================================================
# HELPERS
# ============================================================
def hash_row(rs, teks) -> str:
    s = f"{str(rs).strip().lower()}|{str(teks).strip()}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def pilih_teks(row) -> str:
    for k in [KOLOM_TEKS, "teks"]:
        if k in row and pd.notna(row[k]) and str(row[k]).strip():
            return str(row[k])[:MAX_CHARS]
    return ""

def panggil_llm(rows):
    """rows: list of (hash, teks). Return dict {hash: label}."""
    payload = [{"id": i, "teks": t} for i, (_, t) in enumerate(rows)]
    for percobaan in range(6):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
            hasil = json.loads(r.choices[0].message.content)["hasil"]
            out = {}
            for h in hasil:
                if h.get("label") in VALID:
                    idx = int(h["id"])
                    if 0 <= idx < len(rows):
                        out[rows[idx][0]] = h["label"]
            return out
        except Exception as e:
            wait = min(60, 5 * 2 ** percobaan)
            print(f"  [retry {percobaan+1}/6] {e} -> {wait}s")
            time.sleep(wait)
    return {}

# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_csv(DATA)
    print(f"[INFO] {len(df)} baris dari {DATA}")

    for k in [KOLOM_LABEL, KOLOM_FINAL]:
        if k not in df.columns:
            df[k] = pd.NA

    # hash tiap baris
    df["_hash"] = [hash_row(r.get("rs", ""), pilih_teks(r)) for _, r in df.iterrows()]

    # load checkpoint
    ckpt = {}
    if os.path.exists(CHECKPOINT):
        c = pd.read_csv(CHECKPOINT)
        ckpt = dict(zip(c["hash"], c["label"]))
        print(f"[INFO] checkpoint: {len(ckpt)} label")

    # pasang dari checkpoint dulu
    for i, row in df.iterrows():
        h = row["_hash"]
        if h in ckpt and pd.isna(df.at[i, KOLOM_LABEL]):
            df.at[i, KOLOM_LABEL] = ckpt[h]

    # yang perlu dilabeli (teks tidak kosong & label masih kosong)
    mask_valid = df["_hash"].map(lambda h: h != hash_row("", ""))
    todo = df[mask_valid & df[KOLOM_LABEL].isna()].copy()
    print(f"[INFO] perlu dilabeli: {len(todo)}")

    # prioritas Mei 2026
    if "tanggal_ulasan" in todo.columns and len(todo):
        todo["tanggal_ulasan"] = pd.to_datetime(todo["tanggal_ulasan"], errors="coerce")
        is_mei = (todo["tanggal_ulasan"].dt.year == 2026) & (todo["tanggal_ulasan"].dt.month == 5)
        todo = pd.concat([todo[is_mei], todo[~is_mei]])
        print(f"[INFO] Mei 2026 di depan: {is_mei.sum()} baris (sisanya menyusul)")

    if len(todo) == 0:
        print("[INFO] Semua sudah berlabel.")
        df[KOLOM_FINAL] = df[KOLOM_LABEL].map(MAP_KAPITAL)
        df = df.drop(columns=["_hash"])
        df.to_csv(DATA, index=False)
        return

    n_ok = 0
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        batch = todo.iloc[i:i+BATCH]
        rows = [(r["_hash"], pilih_teks(r)) for _, r in batch.iterrows()]
        hasil = panggil_llm(rows)

        for h, label in hasil.items():
            for ix in df.index[df["_hash"] == h]:
                df.at[ix, KOLOM_LABEL] = label
        n_ok += len(hasil)

        if hasil:
            pd.DataFrame(list(hasil.items()), columns=["hash", "label"]).to_csv(
                CHECKPOINT, mode="a",
                header=not os.path.exists(CHECKPOINT), index=False)

        # progres + ETA
        elapsed = time.time() - t0
        done = i + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - done) / rate / 60 if rate > 0 else 0
        print(f"  batch {i//BATCH+1}: {len(hasil)}/{len(batch)} "
              f"| total {n_ok}/{len(todo)} | ETA {eta:.1f} menit")
        time.sleep(SLEEP)

    # label_final + simpan
    df[KOLOM_FINAL] = df[KOLOM_LABEL].map(MAP_KAPITAL)
    df = df.drop(columns=["_hash"])
    df.to_csv(DATA, index=False)

    print(f"\n[DONE] berlabel: {df[KOLOM_LABEL].notna().sum()}/{len(df)} -> {DATA}")
    print("\nDistribusi label_llm:")
    print(df[KOLOM_LABEL].value_counts(dropna=False).to_string())

    if "tanggal_ulasan" in df.columns:
        df["tanggal_ulasan"] = pd.to_datetime(df["tanggal_ulasan"], errors="coerce")
        print("\nPer tahun:")
        print(df["tanggal_ulasan"].dt.year.value_counts().sort_index().to_string())

if __name__ == "__main__":
    main()