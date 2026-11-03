import pandas as pd
from statsmodels.stats.proportion import proportion_confint

d = pd.read_csv("label_negatif_60.csv")
d = d[d["label_manual"].notna() & (d["label_manual"].astype(str).str.strip() != "")]

n = len(d)
x = (d["label_manual"].str.lower().str.strip() == "keluhan").sum()

print(f"Diperiksa: {n}")
print(f"Keluhan  : {x} ({x/n:.1%})")
print()
print(d["label_manual"].value_counts())

if n < 288:
    lo, hi = proportion_confint(x, n, method="wilson")
    print(f"\nCI 95% Wilson: {lo:.1%} – {hi:.1%}")
else:
    print("\n(Sensus semua 288 — tidak perlu CI)")