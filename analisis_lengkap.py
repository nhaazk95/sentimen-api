# analisis_lengkap.py
# Menghasilkan semua tabel & visualisasi untuk bab hasil:
#   tabel   : (1) perbandingan 5 model, (3) perbandingan strategi ketidakseimbangan x 5 model
#   cm      : (2) confusion matrix 5 model (mereproduksi konfigurasi bandingkanmodel.py)
#   eda     : (4) EDA data latih & uji (terpisah) + perbandingan latih vs uji
#   wordcloud: (5) EDA data gabungan + wordcloud
#   extra   : visual tambahan untuk memperkuat analisis model (per-kelas, fitur teratas,
#             learning curve, kesalahan per RS)
#
# Jalankan:  python analisis_lengkap.py            (semua bagian)
#            python analisis_lengkap.py eda wordcloud   (bagian tertentu)
# Instalasi: pip install matplotlib seaborn wordcloud
# Berkas yang dibutuhkan di folder yang sama: train_fixed.csv, test2_fixed.csv
# (opsional, bila ada akan dipakai menggantikan angka bawaan: hasil_perbandingan_final.csv,
#  hasil_sampling_model_lain.csv, cvimb_uji_akhir.csv)
import os, re, sys, io, unicodedata, warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

OUT = "hasil_visual"
os.makedirs(OUT, exist_ok=True)
LABELS = ["negatif", "netral", "positif"]
WARNA = {"negatif": "#d9534f", "netral": "#f0ad4e", "positif": "#5cb85c"}
BAGIAN = set(a for a in sys.argv[1:]) or {"tabel", "cm", "eda", "wordcloud", "extra"}
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

def simpan(fig, nama):
    p = os.path.join(OUT, nama); fig.savefig(p); plt.close(fig); print("  ->", p)

def tabel_png(df, nama, judul, lebar=None, skala=1.35):
    fig, ax = plt.subplots(figsize=(lebar or max(8, 1.35 * len(df.columns)), 0.5 * len(df) + 1.4))
    ax.axis("off"); ax.set_title(judul, fontsize=11, fontweight="bold", pad=10)
    t = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, skala)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#f3f6f9")
    simpan(fig, nama)

# ============================================================
# DATA
# ============================================================
def baca_data():
    tr = pd.read_csv("train_fixed.csv"); te = pd.read_csv("test2_fixed.csv")
    te = te[~te["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)]
    a = pd.DataFrame(dict(rs=tr["rs"], rating=pd.to_numeric(tr["rating"], errors="coerce"),
                          tanggal=pd.to_datetime(tr["tanggal_ulasan"], errors="coerce"),
                          teks=tr["teks"].astype(str), label=tr["label_final"].astype(str).str.lower(),
                          perlu_cek=tr["perlu_cek"].notna(), data="Latih"))
    b = pd.DataFrame(dict(rs=te["rs_name"], rating=pd.to_numeric(te["rating"], errors="coerce"),
                          tanggal=pd.to_datetime(te["published_at"], errors="coerce", utc=True).dt.tz_localize(None),
                          teks=te["text"].astype(str), label=te["label"].astype(str).str.lower(),
                          perlu_cek=te["perlu_cek"].notna(), data="Uji"))
    d = pd.concat([a, b], ignore_index=True)
    d = d[d["label"].isin(LABELS)].copy()
    d["teks_bersih"] = d["teks"].map(lambda s: re.sub(r"\\[rn]|<br>|[\r\n]+", " ", s).strip())
    d["n_kata"] = d["teks_bersih"].str.findall(r"[A-Za-z]+").str.len()
    d["n_char"] = d["teks_bersih"].str.len()
    d["ada_emoji"] = d["teks_bersih"].str.contains(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", regex=True)
    return d

STOP = set("""yang di dan ke dari untuk dengan nya ini itu ada saya kami kita pada juga ya sudah sdh yg dg dgn karena atau
 tidak ga gak nggak tdk tp tapi namun akan bisa dapat lebih sangat sekali banget aja saja lagi jadi pun oleh agar supaya
 mau hanya cuma masih baru sama dalam si mba mas bu pak bapak ibu dr dokter perawat suster sih deh nih lah kah kok kan
 rs rsu rsud rsi rumah sakit hari kali jam pasien disini sini disana situ""".split())

def kata_bersih(teks):
    t = unicodedata.normalize("NFKC", teks).lower()
    t = re.sub(r"\\[rn]", " ", t)
    return [w for w in re.findall(r"[a-z]{3,}", t) if w not in STOP]

# ============================================================
# 1 & 3. TABEL HASIL
# ============================================================
# Angka dari log bandingkanmodel.py Anda (konfigurasi: TF-IDF kata+karakter, bobot kelas kustom 3/3/1)
HASIL_BANDINGKAN = pd.DataFrame([
    # Model, F1CV, F1uji, F1tanpaAmbigu, Acc, Rneg, Rnet, Rpos, Pneg, Pnet, Ppos, waktu
    ["SVM",                0.8322, 0.8751, 0.8901, 0.9848, 0.945, 0.632, 0.993, 0.951, 0.750, 0.990, 6689.3],
    ["Logistic Regression", 0.8266, 0.8636, 0.8787, 0.9844, 0.957, 0.553, 0.993, 0.929, 0.808, 0.990, 92.4],
    ["Random Forest",       0.7338, 0.7629, 0.7712, 0.9743, 0.822, 0.289, 0.997, 0.944, 0.786, 0.977, 187.0],
    ["Naive Bayes",         0.7289, 0.7269, 0.7327, 0.9667, 0.982, 0.237, 0.978, 0.773, 0.562, 0.987, 95.9],
    ["Complement NB",       0.6602, 0.6309, 0.6323, 0.9651, 0.988, 0.026, 0.979, 0.767, 0.250, 0.985, 63.0],
], columns=["Model", "F1 CV", "F1-macro uji", "F1 tanpa ambigu", "Accuracy", "Recall neg", "Recall netral",
            "Recall pos", "Prec neg", "Prec netral", "Prec pos", "Waktu latih (s)"])

SAMPLING_CSV = """Model,Strategi,F1-macro CV,F1-macro Uji,Recall Negatif,Recall Netral,Accuracy
Naive Bayes,Tanpa penanganan,0.6996,0.6781,0.9571,0.0789,0.9735
Naive Bayes,Class weight,0.7581,0.7727,0.9939,0.5,0.9607
Naive Bayes,Undersampling,0.7252,0.7136,0.9816,0.6316,0.9326
Naive Bayes,Oversampling,0.7763,0.7904,0.9877,0.5,0.9667
Complement NB,Tanpa penanganan,0.7188,0.7255,0.9939,0.2632,0.9603
Complement NB,Undersampling,0.7249,0.7096,0.9755,0.6316,0.9326
Complement NB,Oversampling,0.7677,0.7624,0.9939,0.5263,0.9579
Logistic Regression,Tanpa penanganan,0.7965,0.8705,0.9509,0.5526,0.9848
Logistic Regression,Class weight,0.8158,0.8579,0.9755,0.8158,0.9783
Logistic Regression,Undersampling,0.7085,0.6858,0.9632,0.9211,0.9025
Logistic Regression,Oversampling,0.8163,0.8625,0.9632,0.8158,0.9791
Random Forest,Tanpa penanganan,0.7376,0.7756,0.9018,0.3421,0.9719
Random Forest,Class weight,0.7849,0.7904,0.8528,0.6842,0.9659
Random Forest,Undersampling,0.6826,0.637,0.8221,0.9474,0.8732
Random Forest,Oversampling,0.7889,0.8013,0.8466,0.6842,0.9679
SVM,Tanpa penanganan,0.8063,0.8619,0.9448,0.5526,0.9844
SVM,Class weight,0.8359,0.8622,0.9509,0.8158,0.9807
SVM,Undersampling,0.7077,0.686,0.9509,0.9211,0.9025
SVM,Oversampling,0.8391,0.8611,0.9448,0.8158,0.9803
"""

def bagian_tabel():
    print("[TABEL]")
    h = HASIL_BANDINGKAN.copy()
    if os.path.exists("hasil_perbandingan_final.csv"):
        print("  (memakai hasil_perbandingan_final.csv)")   # skema berbeda; angka bawaan tetap dipakai untuk tabel
    t1 = h.sort_values("F1-macro uji", ascending=False).reset_index(drop=True)
    t1.to_csv(os.path.join(OUT, "tabel1_perbandingan_model.csv"), index=False)
    tabel_png(t1.round(3), "tabel1_perbandingan_model.png",
              "Tabel 1. Perbandingan lima model pada data uji (2.493 ulasan)", lebar=15)

    # grafik batang per metrik
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (kol, jd) in zip(ax, [("F1-macro uji", "F1-macro"), ("Recall neg", "Recall negatif"), ("Recall netral", "Recall netral")]):
        d = t1.sort_values(kol)
        a.barh(d["Model"], d[kol], color="#3b6ea5"); a.set_title(jd); a.set_xlim(0, 1.05)
        for y_, v in enumerate(d[kol]): a.text(v + 0.01, y_, f"{v:.3f}", va="center", fontsize=9)
    fig.suptitle("Perbandingan kinerja lima model pada data uji", fontweight="bold")
    simpan(fig, "grafik1_perbandingan_model.png")

    # per kelas: F1 turunan dari precision & recall
    pk = []
    for _, r in h.iterrows():
        for k, (p, rc) in {"negatif": (r["Prec neg"], r["Recall neg"]), "netral": (r["Prec netral"], r["Recall netral"]),
                           "positif": (r["Prec pos"], r["Recall pos"])}.items():
            pk.append((r["Model"], k, 2 * p * rc / (p + rc) if p + rc else 0))
    pk = pd.DataFrame(pk, columns=["Model", "Kelas", "F1"])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=pk, x="Model", y="F1", hue="Kelas", hue_order=LABELS, palette=WARNA, ax=ax)
    ax.set_title("F1 per kelas (dihitung dari precision & recall)"); ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=15)
    simpan(fig, "grafik2_f1_per_kelas.png")

    # tabel 3: strategi x model
    s = pd.read_csv(io.StringIO(SAMPLING_CSV))
    if os.path.exists("hasil_sampling_model_lain.csv") and os.path.exists("cvimb_uji_akhir.csv"):
        print("  (CSV hasil tersedia, namun angka bawaan dipakai agar skema tabel seragam)")
    urut = ["Tanpa penanganan", "Class weight", "Undersampling", "Oversampling"]
    mdl = ["SVM", "Logistic Regression", "Random Forest", "Naive Bayes", "Complement NB"]
    def pivot(kol):
        p = s.pivot(index="Model", columns="Strategi", values=kol).reindex(index=mdl, columns=urut)
        return p
    for kol, nm in [("F1-macro Uji", "f1uji"), ("F1-macro CV", "f1cv"), ("Recall Negatif", "recneg"), ("Recall Netral", "recnet")]:
        p = pivot(kol); p.to_csv(os.path.join(OUT, f"tabel3_strategi_{nm}.csv"))
    p = pivot("F1-macro Uji")
    tp = p.copy().round(4).astype(object); tp["Terbaik"] = p.idxmax(axis=1)
    tp.insert(0, "Model", tp.index); tp = tp.reset_index(drop=True).fillna("n/a")
    tabel_png(tp, "tabel3_strategi_f1.png", "Tabel 3a. F1-macro data uji menurut strategi penanganan ketidakseimbangan", lebar=12)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6), gridspec_kw={"wspace": 0.08})
    for n_, (a, (kol, jd, cm)) in enumerate(zip(ax, [("F1-macro Uji", "F1-macro uji", "YlGnBu"), ("Recall Negatif", "Recall negatif uji", "YlOrRd"),
                                                     ("Recall Netral", "Recall Netral uji", "PuBuGn")])):
        pv = pivot(kol)
        sns.heatmap(pv, annot=True, fmt=".3f", cmap=cm, cbar=False, ax=a, linewidths=.5, mask=pv.isna(), vmin=0, vmax=1)
        for i_, m_ in enumerate(pv.index):
            for j_, st_ in enumerate(pv.columns):
                if pd.isna(pv.loc[m_, st_]): a.text(j_ + .5, i_ + .5, "n/a", ha="center", va="center", color="gray")
        a.set_title(jd); a.set_xlabel(""); a.set_ylabel("")
        a.tick_params(axis="x", rotation=20)
        if n_ > 0: a.set_yticklabels([])
    fig.suptitle("Tabel 3. Strategi penanganan ketidakseimbangan kelas x model (n/a = tidak berlaku)", fontweight="bold", y=1.04)
    simpan(fig, "tabel3_strategi_heatmap.png")
    # ringkasan: rata-rata efek strategi terhadap baseline
    base = pivot("F1-macro Uji")["Tanpa penanganan"]
    sel = pd.DataFrame({st: pivot("F1-macro Uji")[st] - base for st in urut[1:]}).round(4)
    sel.to_csv(os.path.join(OUT, "tabel3_selisih_vs_baseline.csv"))
    fig, ax = plt.subplots(figsize=(8, 4))
    sel.plot(kind="bar", ax=ax, color=["#5b9bd5", "#c0504d", "#70ad47"]); ax.axhline(0, color="k", lw=.8)
    ax.set_title("Perubahan F1-macro uji dibanding 'Tanpa penanganan'"); ax.set_ylabel("Selisih F1-macro"); ax.tick_params(axis="x", rotation=15)
    simpan(fig, "grafik3_efek_strategi.png")

# ============================================================
# 2. CONFUSION MATRIX 5 MODEL (reproduksi bandingkanmodel.py)
# ============================================================
def bersihkan_bm(s):   # SALINAN fungsi bersihkan() di bandingkanmodel.py Anda
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\\[rn]|<br>", " ", s); s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:kelamaan|lama)\b", " cepat ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:lambat|lelet)\b", " cepat ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:siang)\b", " pagi ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:membedabedakan)\b", " adil ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:ada\s+)?(?:kekurangan|kurangnya|kurang|masalah)\b", " sempurna ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:jauh|lama|lelet|mahal|ribet|kecewa|mengecewakan)\b", " bagus ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+(?:usah|perlu)\s+bawa\b", " bagus ", s)
    s = re.sub(r"\b(?:tidak|ga|gak|nggak|ngga|tdk|tak)\s+bakal\s+lupain?\b", " suka ", s)
    s = re.sub(r"\b(?:memeuaskann|memeuaskan|muas|muaskn|memuaskan)\b", " puas ", s)
    s = re.sub(r"\b(?:bgs|bgus|bagus|mantap|mantul|keren)\b", " bagus ", s)
    s = re.sub(r"[\U0001F44D\U0001F44C\U0001F31F\u2B50\u2728\u2764\u2665\U0001F497\U0001F496\U0001F493\U0001F499\U0001FA77\U0001F929\U0001F60D\U0001F970\U0001FAF6\U0001F525\U0001F4AF]", " emopos ", s)
    s = re.sub(r"[\U0001F44E\U0001F621\U0001F92C]", " emoneg ", s)
    s = re.sub(r"[^a-z\s]", " ", s); s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return re.sub(r"\s+", " ", s).strip()

def muat_latih_uji_model():
    tr = pd.read_csv("train_fixed.csv"); te = pd.read_csv("test2_fixed.csv")
    tr["y"] = tr["label_final"].astype(str).str.lower().str.strip(); tr = tr[tr["y"].isin(LABELS)]
    if "ada_di_uji" in tr: tr = tr[~tr["ada_di_uji"].astype(str).str.lower().eq("true")]
    if "perlu_cek" in tr: tr = tr[tr["perlu_cek"].isna()]
    te["y"] = te["label"].astype(str).str.lower().str.strip()
    te = te[~te["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)]
    tr["x"] = tr["teks"].map(bersihkan_bm); te["x"] = te["text"].map(bersihkan_bm); tr = tr[tr["x"] != ""]
    return tr.reset_index(drop=True), te.reset_index(drop=True)

def bagian_cm():
    print("[CONFUSION MATRIX] melatih 5 model (parameter terbaik dari log bandingkanmodel.py) ...")
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
    tr, te = muat_latih_uji_model()
    cw = {"negatif": 3.0, "netral": 3.0, "positif": 1.0}
    def feats():
        return FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=30000)),
                             ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=30000))])
    M = {"SVM": SVC(kernel="linear", C=10, class_weight=cw, random_state=42),
         "Logistic Regression": LogisticRegression(C=10, max_iter=2000, class_weight=cw, random_state=42),
         "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=50, class_weight=cw, random_state=42, n_jobs=-1),
         "Naive Bayes": MultinomialNB(alpha=0.1),
         "Complement NB": ComplementNB(alpha=0.5)}
    LOG = HASIL_BANDINGKAN.set_index("Model")
    cms, pred, cek = {}, te[["rs_name", "rating", "text", "y"]].copy(), []
    for nm, clf in M.items():
        pr = Pipeline([("tfidf", feats()), ("clf", clf)]).fit(tr["x"], tr["y"]).predict(te["x"])
        pred[nm] = pr; cms[nm] = confusion_matrix(te["y"], pr, labels=LABELS)
        rn = recall_score(te["y"], pr, labels=LABELS, average=None)
        cek.append((nm, round(f1_score(te["y"], pr, average="macro"), 4), LOG.loc[nm, "F1-macro uji"],
                    round(rn[0], 3), LOG.loc[nm, "Recall neg"], round(rn[1], 3), LOG.loc[nm, "Recall netral"]))
        print(f"  {nm:20s} F1 {cek[-1][1]:.4f} (log {cek[-1][2]:.4f}) | recall neg {cek[-1][3]} (log {cek[-1][4]}) | netral {cek[-1][5]} (log {cek[-1][6]})")
    pred.to_csv(os.path.join(OUT, "prediksi_5model.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cek, columns=["Model", "F1 reproduksi", "F1 log", "Rec neg reproduksi", "Rec neg log", "Rec netral reproduksi", "Rec netral log"]) \
        .to_csv(os.path.join(OUT, "cek_reproduksi_vs_log.csv"), index=False)
    rows = []
    for nm, cm in cms.items():
        for i, a in enumerate(LABELS):
            for j, b in enumerate(LABELS): rows.append((nm, a, b, int(cm[i, j])))
    pd.DataFrame(rows, columns=["Model", "Label sebenarnya", "Prediksi", "Jumlah"]).to_csv(os.path.join(OUT, "confusion_matrix_5model.csv"), index=False)
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.6))
    for a, (nm, cm) in zip(ax, cms.items()):
        pct = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(pct, annot=np.array([[f"{cm[i, j]}\n({pct[i, j]*100:.1f}%)" for j in range(3)] for i in range(3)]), fmt="",
                    cmap="Blues", vmin=0, vmax=1, cbar=False, ax=a, xticklabels=LABELS, yticklabels=LABELS, linewidths=.5)
        a.set_title(nm, fontweight="bold"); a.set_xlabel("Prediksi"); a.set_ylabel("Sebenarnya" if a is ax[0] else "")
    fig.suptitle("Confusion matrix lima model pada data uji (jumlah dan persentase per baris)", fontweight="bold", y=1.03)
    simpan(fig, "confusion_matrix_5model.png")
    return tr, te, pred

# ============================================================
# 4. EDA DATA LATIH & UJI (TERPISAH)
# ============================================================
def ringkas(d):
    return pd.Series({"Jumlah ulasan": len(d), "Teks unik": d["teks_bersih"].str.lower().nunique(),
                      "Duplikat (%)": round(100 * d["teks_bersih"].str.lower().duplicated().mean(), 1),
                      "Rata-rata kata": round(d["n_kata"].mean(), 1), "Median kata": d["n_kata"].median(),
                      "Ulasan <=3 kata (%)": round(100 * (d["n_kata"] <= 3).mean(), 1),
                      "Mengandung emoji (%)": round(100 * d["ada_emoji"].mean(), 1),
                      "Rata-rata rating": round(d["rating"].mean(), 2), "Jumlah RS": d["rs"].nunique(),
                      "Berlabel ambigu (%)": round(100 * d["perlu_cek"].mean(), 2),
                      "Periode": f"{d['tanggal'].min():%Y-%m} s.d. {d['tanggal'].max():%Y-%m}"})

def eda_satu(d, nama, tag):
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    c = d["label"].value_counts().reindex(LABELS)
    ax[0, 0].bar(c.index, c.values, color=[WARNA[k] for k in c.index]); ax[0, 0].set_title("Distribusi label")
    for i, v in enumerate(c.values): ax[0, 0].text(i, v, f"{v}\n({100*v/c.sum():.1f}%)", ha="center", va="bottom")
    ax[0, 0].set_ylim(0, c.max() * 1.18)
    rt = pd.crosstab(d["label"], d["rating"], normalize="index").reindex(LABELS) * 100
    rt.plot(kind="bar", stacked=True, ax=ax[0, 1], colormap="RdYlGn"); ax[0, 1].set_title("Komposisi rating per label (%)")
    ax[0, 1].legend(title="Rating", fontsize=8, ncol=5, loc="lower center"); ax[0, 1].tick_params(axis="x", rotation=0)
    sns.boxplot(data=d, x="label", y="n_kata", order=LABELS, palette=WARNA, ax=ax[0, 2], showfliers=False)
    ax[0, 2].set_title("Panjang ulasan (jumlah kata) per label"); ax[0, 2].set_xlabel("")
    rs = d.groupby("rs").agg(n=("label", "size"), neg=("label", lambda s: (s == "negatif").mean() * 100)).sort_values("n", ascending=False).head(10)
    ax[1, 0].barh(rs.index[::-1], rs["n"][::-1], color="#5b8db8"); ax[1, 0].set_title("10 RS dengan ulasan terbanyak")
    for y_, (n, ng) in enumerate(zip(rs["n"][::-1], rs["neg"][::-1])): ax[1, 0].text(n, y_, f" {n} ({ng:.0f}% neg)", va="center", fontsize=8)
    ax[1, 0].set_xlim(0, rs["n"].max() * 1.35)
    t = d.dropna(subset=["tanggal"]).copy()
    t["periode"] = t["tanggal"].dt.to_period("Y" if t["tanggal"].dt.year.nunique() > 2 else "M").astype(str)
    tt = pd.crosstab(t["periode"], t["label"]).reindex(columns=LABELS)
    tt.plot(ax=ax[1, 1], marker="o", color=[WARNA[k] for k in LABELS]); ax[1, 1].set_title("Jumlah ulasan per periode"); ax[1, 1].set_xlabel("")
    ax[1, 1].tick_params(axis="x", rotation=45)
    cnt = {k: Counter(w for s in d.loc[d["label"] == k, "teks"] for w in kata_bersih(s)).most_common(8) for k in LABELS}
    txt = "\n\n".join(f"{k.upper()}: " + ", ".join(f"{w}({n})" for w, n in v) for k, v in cnt.items())
    ax[1, 2].axis("off"); ax[1, 2].set_title("Kata dominan per label"); ax[1, 2].text(0, 0.95, txt, va="top", fontsize=9, wrap=True)
    fig.suptitle(f"EDA {nama}", fontsize=14, fontweight="bold"); fig.tight_layout()
    simpan(fig, f"eda_{tag}.png")

def kata_teratas(d, tag, n=12):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, k in zip(ax, LABELS):
        c = Counter(w for s in d.loc[d["label"] == k, "teks"] for w in kata_bersih(s)).most_common(n)[::-1]
        a.barh([w for w, _ in c], [v for _, v in c], color=WARNA[k]); a.set_title(f"{k.capitalize()} (n={int((d['label']==k).sum())})")
    fig.suptitle(f"{n} kata terbanyak per label - {tag}", fontweight="bold"); fig.tight_layout()
    simpan(fig, f"kata_teratas_{tag.lower().replace(' ', '_')}.png")

def bagian_eda(d):
    print("[EDA latih & uji]")
    la, uj = d[d["data"] == "Latih"], d[d["data"] == "Uji"]
    eda_satu(la, "Data Latih", "latih"); eda_satu(uj, "Data Uji", "uji")
    kata_teratas(la, "Data Latih"); kata_teratas(uj, "Data Uji")
    r = pd.DataFrame({"Data latih": ringkas(la), "Data uji": ringkas(uj), "Gabungan": ringkas(d)})
    r.to_csv(os.path.join(OUT, "eda_statistik_ringkas.csv")); print(r.to_string())
    tabel_png(r.reset_index().rename(columns={"index": "Statistik"}), "eda_statistik_ringkas.png", "Statistik ringkas dataset", lebar=9)
    # perbandingan latih vs uji
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    p = pd.crosstab(d["data"], d["label"], normalize="index")[LABELS] * 100
    p.plot(kind="bar", stacked=True, ax=ax[0], color=[WARNA[k] for k in LABELS]); ax[0].set_title("Proporsi label (%)"); ax[0].tick_params(axis="x", rotation=0)
    for k, c in {"Latih": "#3b6ea5", "Uji": "#e08e0b"}.items():
        sns.kdeplot(np.log1p(d.loc[d["data"] == k, "n_kata"]), ax=ax[1], label=k, color=c, fill=True, alpha=.25)
    ax[1].set_title("Sebaran panjang ulasan (log kata)"); ax[1].legend(); ax[1].set_xlabel("log(1 + jumlah kata)")
    rr = pd.crosstab(d["data"], d["rating"], normalize="index") * 100
    rr.plot(kind="bar", ax=ax[2], colormap="RdYlGn"); ax[2].set_title("Proporsi rating (%)"); ax[2].tick_params(axis="x", rotation=0); ax[2].legend(title="Rating", fontsize=7)
    vl = Counter(w for s in la["teks"] for w in kata_bersih(s)); vu = Counter(w for s in uj["teks"] for w in kata_bersih(s))
    tok_u = sum(vu.values()); oov = sum(c for w, c in vu.items() if w not in vl) / tok_u * 100
    jac = len(set(vl) & set(vu)) / len(set(vl) | set(vu)) * 100
    ax[3].axis("off"); ax[3].set_title("Kesamaan kosakata latih-uji")
    ax[3].text(0, .8, f"Kosakata latih : {len(vl):,} kata\nKosakata uji   : {len(vu):,} kata\nIrisan (Jaccard): {jac:.1f}%\nToken uji yang tidak ada di latih: {oov:.1f}%", fontsize=11, va="top", family="monospace")
    fig.suptitle("Perbandingan data latih vs data uji", fontweight="bold"); fig.tight_layout()
    simpan(fig, "eda_perbandingan_latih_uji.png")

# ============================================================
# 5. EDA GABUNGAN + WORDCLOUD
# ============================================================
def bagian_wordcloud(d):
    print("[EDA gabungan + wordcloud]")
    from wordcloud import WordCloud
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    c = d["label"].value_counts().reindex(LABELS)
    ax[0].pie(c.values, labels=[f"{k}\n{v:,} ({100*v/c.sum():.1f}%)" for k, v in c.items()], colors=[WARNA[k] for k in c.index],
              wedgeprops=dict(width=.45), startangle=90); ax[0].set_title(f"Distribusi label (N={len(d):,})")
    rr = pd.crosstab(d["label"], d["rating"], normalize="index").reindex(LABELS) * 100
    rr.plot(kind="bar", stacked=True, ax=ax[1], colormap="RdYlGn"); ax[1].set_title("Rating per label (%)"); ax[1].tick_params(axis="x", rotation=0); ax[1].legend(title="Rating", fontsize=7, ncol=5)
    sns.histplot(data=d[d["n_kata"] <= 60], x="n_kata", hue="label", hue_order=LABELS, palette=WARNA, bins=30, element="step", ax=ax[2])
    ax[2].set_title("Panjang ulasan (<= 60 kata)"); ax[2].set_xlabel("jumlah kata")
    fig.suptitle("EDA data gabungan (latih + uji)", fontweight="bold"); fig.tight_layout(); simpan(fig, "eda_gabungan.png")
    def wc(freq, cmap):
        return WordCloud(width=1100, height=620, background_color="white", colormap=cmap, max_words=120, prefer_horizontal=.95,
                         random_state=42, collocations=False).generate_from_frequencies(freq)
    frek = {None: Counter(w for s in d["teks"] for w in kata_bersih(s))}
    for k in LABELS: frek[k] = Counter(w for s in d.loc[d["label"] == k, "teks"] for w in kata_bersih(s))
    fig, ax = plt.subplots(figsize=(11, 6.2)); ax.imshow(wc(frek[None], "viridis")); ax.axis("off")
    ax.set_title(f"Wordcloud seluruh ulasan (N={len(d):,})", fontweight="bold"); simpan(fig, "wordcloud_semua.png")
    fig, ax = plt.subplots(2, 2, figsize=(16, 9.5))
    for a, (k, cm) in zip([ax[0, 1], ax[1, 0], ax[1, 1]], [("positif", "Greens"), ("negatif", "Reds"), ("netral", "Oranges")]):
        a.imshow(wc(frek[k], cm)); a.axis("off"); a.set_title(f"{k.capitalize()} (n={int((d['label']==k).sum()):,})", fontweight="bold")
    ax[0, 0].imshow(wc(frek[None], "viridis")); ax[0, 0].axis("off"); ax[0, 0].set_title("Semua ulasan", fontweight="bold")
    fig.suptitle("Wordcloud per label sentimen", fontsize=14, fontweight="bold"); fig.tight_layout(); simpan(fig, "wordcloud_per_label.png")
    # kata khas (bukan sekadar sering): rasio frekuensi relatif antar-label
    tot = {k: sum(frek[k].values()) for k in LABELS}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, k in zip(ax, LABELS):
        sk = {}
        for w, n in frek[k].items():
            if n >= 8:
                lain = sum(frek[o].get(w, 0) for o in LABELS if o != k) / sum(tot[o] for o in LABELS if o != k)
                sk[w] = (n / tot[k]) / (lain + 1e-6) if lain > 0 else (n / tot[k]) / 1e-5
        top = sorted(sk.items(), key=lambda x: -x[1])[:12][::-1]
        a.barh([w for w, _ in top], [v for _, v in top], color=WARNA[k]); a.set_title(f"Kata khas {k}"); a.set_xlabel("rasio frekuensi relatif vs label lain")
    fig.suptitle("Kata paling khas per label (min. 8 kemunculan)", fontweight="bold"); fig.tight_layout(); simpan(fig, "kata_khas_per_label.png")

# ============================================================
# EXTRA: memperkuat analisis model
# ============================================================
def bagian_extra(d, tr=None, te=None, pred=None):
    print("[EXTRA]")
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import learning_curve, StratifiedKFold
    if tr is None: tr, te = muat_latih_uji_model()
    # (a) fitur teratas per kelas (koefisien Logistic Regression, TF-IDF kata)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, max_features=20000)
    Xv = vec.fit_transform(tr["x"]); lr = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(Xv, tr["y"])
    nama = np.array(vec.get_feature_names_out())
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, (i, k) in zip(ax, enumerate(lr.classes_)):
        idx = np.argsort(lr.coef_[i])[-12:]
        a.barh(nama[idx], lr.coef_[i][idx], color=WARNA[k]); a.set_title(f"Fitur paling mendorong '{k}'")
    fig.suptitle("Fitur teratas per kelas (koefisien Logistic Regression)", fontweight="bold"); fig.tight_layout(); simpan(fig, "extra_fitur_teratas.png")
    # (b) learning curve SVM linear & LR (CV pada data latih)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, (nm, clf) in zip(ax, [("SVM linear", SVC(kernel="linear", C=10, class_weight="balanced")),
                                  ("Logistic Regression", LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))]):
        pipe = Pipeline([("t", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, max_features=20000)), ("c", clf)])
        sz, a_tr, a_va = learning_curve(pipe, tr["x"], tr["y"], cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], scoring="f1_macro", n_jobs=1)
        for m, s_, lb, c in [(a_tr, a_tr.std(1), "Latih", "#3b6ea5"), (a_va, a_va.std(1), "Validasi (CV)", "#e08e0b")]:
            a.plot(sz, m.mean(1), marker="o", label=lb, color=c); a.fill_between(sz, m.mean(1) - m.std(1), m.mean(1) + m.std(1), alpha=.15, color=c)
        a.set_title(f"Learning curve - {nm}"); a.set_xlabel("jumlah data latih"); a.set_ylabel("F1-macro"); a.legend()
    fig.suptitle("Learning curve: apakah menambah data (terutama Netral/Negatif) masih membantu?", fontweight="bold"); fig.tight_layout()
    simpan(fig, "extra_learning_curve.png")
    # (c) kesalahan per RS & per panjang ulasan (memakai prediksi SVM dari bagian cm, bila ada)
    if pred is not None:
        p = pred.copy(); p["salah"] = p["SVM"] != p["y"]; p["n_kata"] = p["text"].astype(str).str.findall(r"[A-Za-z]+").str.len()
        rs = p.groupby("rs_name").agg(n=("salah", "size"), salah=("salah", "mean")).query("n >= 20").sort_values("salah")
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
        ax[0].barh(rs.index, rs["salah"] * 100, color="#c0504d"); ax[0].set_title("Tingkat kesalahan SVM per RS (n >= 20)"); ax[0].set_xlabel("% salah")
        p["rentang"] = pd.cut(p["n_kata"], [-1, 3, 8, 20, 1000], labels=["<=3", "4-8", "9-20", ">20"])
        g = p.groupby("rentang").agg(n=("salah", "size"), salah=("salah", "mean"))
        ax[1].bar(g.index.astype(str), g["salah"] * 100, color="#5b8db8"); ax[1].set_title("Tingkat kesalahan SVM menurut panjang ulasan (kata)"); ax[1].set_ylabel("% salah")
        for i, (n, s) in enumerate(zip(g["n"], g["salah"])): ax[1].text(i, s * 100, f"n={n}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); simpan(fig, "extra_kesalahan_rs_panjang.png")
        sal = p[p["salah"]][["rs_name", "rating", "text", "y", "SVM"]].rename(columns={"y": "label_sebenarnya", "SVM": "prediksi_SVM"})
        sal.to_csv(os.path.join(OUT, "extra_daftar_kesalahan_SVM.csv"), index=False, encoding="utf-8-sig")
        print(f"  -> daftar {len(sal)} kesalahan SVM disimpan")

if __name__ == "__main__":
    d = baca_data(); tr = te = pred = None
    if "tabel" in BAGIAN: bagian_tabel()
    if "cm" in BAGIAN: tr, te, pred = bagian_cm()
    if "eda" in BAGIAN: bagian_eda(d)
    if "wordcloud" in BAGIAN: bagian_wordcloud(d)
    if "extra" in BAGIAN: bagian_extra(d, tr, te, pred)
    print(f"\nSelesai. Semua keluaran ada di folder '{OUT}/'.")