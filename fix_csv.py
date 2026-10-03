import re
import sys

import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "train_pool.csv"
DST = sys.argv[2] if len(sys.argv) > 2 else "train_pool_fixed.csv"
end_re = re.compile(r",\s*(Positif|Negatif|Netral)\s*$")

with open(SRC, encoding="utf-8-sig", newline="") as f:
    lines = f.read().replace("\r\n", "\n").split("\n")

records, buf = [], []
for line in lines[1:]:                      # lewati header
    buf.append(line)
    if end_re.search(line):                 # baris berakhir dengan label -> record selesai
        full = "\n".join(buf)
        text, label = full.rsplit(",", 1)   # label = setelah koma TERAKHIR
        text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
        records.append((text, label.strip()))
        buf = []

dropped = [b for b in buf if b.strip()]
df = pd.DataFrame(records, columns=["text", "label"])
df = df[df["text"] != ""]
df.to_csv(DST, index=False, quoting=1, encoding="utf-8")   # QUOTE_ALL

print("records:", len(df), "| sisa baris tanpa label (dibuang):", len(dropped))
print(df["label"].value_counts())