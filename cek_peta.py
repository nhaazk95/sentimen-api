import pandas as pd
df = pd.read_json("https://sentimen-api.onrender.com/reviews.json")
t = (df.groupby(["Nama RS", "Sentimen_Prediksi"]).size()
       .unstack(fill_value=0))
t["total"] = t.sum(axis=1)
t["% negatif"] = (100 * t.get("Negatif", 0) / t["total"]).round(1)
t["sampel kecil"] = t["total"] < 20
print(t.sort_values("total", ascending=False).to_string())
t.to_csv("peta_sentimen_per_rs.csv")