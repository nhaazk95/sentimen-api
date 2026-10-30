import joblib
m = joblib.load("svm_pipeline_final.pkl")            # sesuaikan nama modelmu
unik = unik.dropna(subset=["clean","rs"])
unik["pred"] = m.predict(unik["clean"])
tab = unik.groupby(["rs","pred"]).size().unstack(fill_value=0)
tab["total"] = tab.sum(axis=1)
print(tab.sort_values("total", ascending=False))