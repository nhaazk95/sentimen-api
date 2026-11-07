# analisis_lengkap.py
# Semua gambar disimpan sebagai file TERPISAH (satu panel = satu PNG).
# Jalankan:  python analisis_lengkap.py            (semua bagian)
#            python analisis_lengkap.py eda wordcloud   (bagian tertentu)
# Instalasi: pip install matplotlib seaborn wordcloud
# Berkas yang dibutuhkan di folder yang sama: train_fixed.csv, test2_fixed.csv

import os
import re
import sys
import io
import textwrap
import unicodedata
import warnings
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
BAGIAN = set(sys.argv[1:]) or {"tabel", "cm", "eda", "wordcloud", "extra"}
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})


# ------------------------------------------------------------------ util
def simpan(fig, nama):
    """Selalu tight_layout dulu sebelum savefig, supaya tidak ada yang numpuk."""
    try:
        fig.tight_layout()
    except Exception:
        pass
    p = os.path.join(OUT, nama)
    fig.savefig(p)
    plt.close(fig)
    print("  ->", p)


def wrap_label(s, n=30):
    return textwrap.fill(str(s), n)


def tabel_png(df, nama, judul, lebar=None, skala=1.35):
    fig, ax = plt.subplots(figsize=(lebar or max(8, 1.35 * len(df.columns)),
                                    0.5 * len(df) + 1.4))
    ax.axis("off")
    ax.set_title(wrap_label(judul, 90), fontsize=11, fontweight="bold", pad=10)
    t = ax.table(cellText=df.values, colLabels=df.columns,
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, skala)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f6f9")
    simpan(fig, nama)


# ------------------------------------------------------------------ data
def baca_data():
    tr = pd.read_csv("train_fixed.csv")
    te = pd.read_csv("test2_fixed.csv")
    te = te[~te["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)]
    a = pd.DataFrame(dict(
        rs=tr["rs"],
        rating=pd.to_numeric(tr["rating"], errors="coerce"),
        tanggal=pd.to_datetime(tr["tanggal_ulasan"], errors="coerce"),
        teks=tr["teks"].astype(str),
        label=tr["label_final"].astype(str).str.lower(),
        perlu_cek=tr["perlu_cek"].notna(),
        data="Latih"))
    b = pd.DataFrame(dict(
        rs=te["rs_name"],
        rating=pd.to_numeric(te["rating"], errors="coerce"),
        tanggal=pd.to_datetime(te["published_at"], errors="coerce", utc=True)
        .dt.tz_localize(None),
        teks=te["text"].astype(str),
        label=te["label"].astype(str).str.lower(),
        perlu_cek=te["perlu_cek"].notna(),
        data="Uji"))
    d = pd.concat([a, b], ignore_index=True)
    d = d[d["label"].isin(LABELS)].copy()
    d["teks_bersih"] = d["teks"].map(
        lambda s: re.sub(r"\\[rn]|<br>|[\r\n]+", " ", s).strip())
    d["n_kata"] = d["teks_bersih"].str.findall(r"[A-Za-z]+").str.len()
    d["n_char"] = d["teks_bersih"].str.len()
    d["ada_emoji"] = d["teks_bersih"].str.contains(
        r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", regex=True)
    return d


STOP = set("""yang di dan ke dari untuk dengan nya ini itu ada saya kami kita pada juga ya sudah sdh yg dg dgn karena atau
 tidak ga gak nggak tdk tp tapi namun akan bisa dapat lebih sangat sekali banget aja saja lagi jadi pun oleh agar supaya
 mau hanya cuma masih baru sama dalam si mba mas bu pak bapak ibu dr dokter perawat suster sih deh nih lah kah kok kan
 rs rsu rsud rsi rumah sakit hari kali jam pasien disini sini disana situ""".split())


def kata_bersih(teks):
    t = unicodedata.normalize("NFKC", teks).lower()
    t = re.sub(r"\\[rn]", " ", t)
    return [w for w in re.findall(r"[a-z]{3,}", t) if w not in STOP]


# ------------------------------------------------------------------ 1 & 3. tabel
HASIL_BANDINGKAN = pd.DataFrame([
    ["SVM",                 0.8322, 0.8751, 0.8901, 0.9848, 0.945, 0.632, 0.993, 0.951, 0.750, 0.990, 6689.3],
    ["Logistic Regression", 0.8266, 0.8636, 0.8787, 0.9844, 0.957, 0.553, 0.993, 0.929, 0.808, 0.990, 92.4],
    ["Random Forest",       0.7338, 0.7629, 0.7712, 0.9743, 0.822, 0.289, 0.997, 0.944, 0.786, 0.977, 187.0],
    ["Naive Bayes",         0.7289, 0.7269, 0.7327, 0.9667, 0.982, 0.237, 0.978, 0.773, 0.562, 0.987, 95.9],
    ["Complement NB",       0.6602, 0.6309, 0.6323, 0.9651, 0.988, 0.026, 0.979, 0.767, 0.250, 0.985, 63.0],
], columns=["Model", "F1 CV", "F1-macro uji", "F1 tanpa ambigu", "Accuracy",
            "Recall neg", "Recall netral", "Recall pos",
            "Prec neg", "Prec netral", "Prec pos", "Waktu latih (s)"])

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
    t1 = h.sort_values("F1-macro uji", ascending=False).reset_index(drop=True)
    t1.to_csv(os.path.join(OUT, "tabel1_perbandingan_model.csv"), index=False)
    tabel_png(t1.round(3), "tabel1_perbandingan_model.png",
              "Tabel 1. Perbandingan lima model pada data uji (2.493 ulasan)",
              lebar=15)

    # grafik batang per metrik -> SATU FILE PER METRIK
    for kol, jd, nm in [("F1-macro uji", "F1-macro", "f1_macro"),
                        ("Recall neg", "Recall negatif", "recall_neg"),
                        ("Recall netral", "Recall netral", "recall_netral")]:
        d = t1.sort_values(kol)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(d["Model"], d[kol], color="#3b6ea5")
        ax.set_title(f"Perbandingan {jd} lima model (data uji)")
        ax.set_xlim(0, 1.08)
        for y_, v in enumerate(d[kol]):
            ax.text(v + 0.01, y_, f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel(jd)
        simpan(fig, f"grafik1_{nm}.png")

    # per kelas: F1 turunan dari precision & recall
    pk = []
    for _, r in h.iterrows():
        for k, (p, rc) in {
            "negatif": (r["Prec neg"], r["Recall neg"]),
            "netral": (r["Prec netral"], r["Recall netral"]),
            "positif": (r["Prec pos"], r["Recall pos"]),
        }.items():
            pk.append((r["Model"], k, 2 * p * rc / (p + rc) if p + rc else 0))
    pk = pd.DataFrame(pk, columns=["Model", "Kelas", "F1"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(data=pk, x="Model", y="F1", hue="Kelas",
                hue_order=LABELS, palette=WARNA, ax=ax)
    ax.set_title("F1 per kelas (dihitung dari precision & recall)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.legend(title="Kelas", bbox_to_anchor=(1.02, 1), loc="upper left")
    simpan(fig, "grafik2_f1_per_kelas.png")

    # tabel 3
    s = pd.read_csv(io.StringIO(SAMPLING_CSV))
    urut = ["Tanpa penanganan", "Class weight", "Undersampling", "Oversampling"]
    mdl = ["SVM", "Logistic Regression", "Random Forest",
           "Naive Bayes", "Complement NB"]

    def pivot(kol):
        return s.pivot(index="Model", columns="Strategi", values=kol) \
                .reindex(index=mdl, columns=urut)

    for kol, nm in [("F1-macro Uji", "f1uji"), ("F1-macro CV", "f1cv"),
                    ("Recall Negatif", "recneg"), ("Recall Netral", "recnet")]:
        pivot(kol).to_csv(os.path.join(OUT, f"tabel3_strategi_{nm}.csv"))

    p = pivot("F1-macro Uji")
    tp = p.copy().round(4).astype(object)
    tp["Terbaik"] = p.idxmax(axis=1)
    tp.insert(0, "Model", tp.index)
    tp = tp.reset_index(drop=True).fillna("n/a")
    tabel_png(tp, "tabel3_strategi_f1.png",
              "Tabel 3a. F1-macro data uji menurut strategi penanganan "
              "ketidakseimbangan", lebar=12)

    # heatmap per metrik -> SATU FILE PER METRIK
    for kol, jd, cm in [("F1-macro Uji", "F1-macro uji", "YlGnBu"),
                        ("Recall Negatif", "Recall negatif uji", "YlOrRd"),
                        ("Recall Netral", "Recall netral uji", "PuBuGn")]:
        pv = pivot(kol)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sns.heatmap(pv, annot=True, fmt=".3f", cmap=cm, cbar=True, ax=ax,
                    linewidths=.5, mask=pv.isna(), vmin=0, vmax=1,
                    annot_kws={"fontsize": 10})
        for i_, m_ in enumerate(pv.index):
            for j_, st_ in enumerate(pv.columns):
                if pd.isna(pv.loc[m_, st_]):
                    ax.text(j_ + .5, i_ + .5, "n/a", ha="center",
                            va="center", color="gray")
        ax.set_title(f"Strategi × model — {jd}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.tick_params(axis="y", rotation=0)
        simpan(fig, f"tabel3_heatmap_{jd.replace(' ', '_')}.png")

    # selisih vs baseline
    base = pivot("F1-macro Uji")["Tanpa penanganan"]
    sel = pd.DataFrame({st: pivot("F1-macro Uji")[st] - base
                        for st in urut[1:]}).round(4)
    sel.to_csv(os.path.join(OUT, "tabel3_selisih_vs_baseline.csv"))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sel.plot(kind="bar", ax=ax, color=["#5b9bd5", "#c0504d", "#70ad47"])
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("Perubahan F1-macro uji dibanding 'Tanpa penanganan'")
    ax.set_ylabel("Selisih F1-macro")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.legend(title="Strategi", bbox_to_anchor=(1.02, 1), loc="upper left")
    simpan(fig, "grafik3_efek_strategi.png")


# ------------------------------------------------------------------ 2. CM
def bersihkan_bm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\\[rn]|<br>", " ", s)
    s = re.sub(r"http\S+", " ", s)
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
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return re.sub(r"\s+", " ", s).strip()


def muat_latih_uji_model():
    tr = pd.read_csv("train_fixed.csv")
    te = pd.read_csv("test2_fixed.csv")
    tr["y"] = tr["label_final"].astype(str).str.lower().str.strip()
    tr = tr[tr["y"].isin(LABELS)]
    if "ada_di_uji" in tr:
        tr = tr[~tr["ada_di_uji"].astype(str).str.lower().eq("true")]
    if "perlu_cek" in tr:
        tr = tr[tr["perlu_cek"].isna()]
    te["y"] = te["label"].astype(str).str.lower().str.strip()
    te = te[~te["text"].astype(str).str.contains(r"\[UJI SISTEM\]", regex=True)]
    tr["x"] = tr["teks"].map(bersihkan_bm)
    te["x"] = te["text"].map(bersihkan_bm)
    tr = tr[tr["x"] != ""]
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def bagian_cm():
    print("[CONFUSION MATRIX] melatih 5 model ...")
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix, f1_score, recall_score

    tr, te = muat_latih_uji_model()
    cw = {"negatif": 3.0, "netral": 3.0, "positif": 1.0}

    def feats():
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                     sublinear_tf=True, max_features=30000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=3, sublinear_tf=True,
                                     max_features=30000))])

    M = {
        "SVM": SVC(kernel="linear", C=10, class_weight=cw, random_state=42),
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000,
                                                  class_weight=cw, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=50,
                                                class_weight=cw, random_state=42,
                                                n_jobs=-1),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Complement NB": ComplementNB(alpha=0.5),
    }
    LOG = HASIL_BANDINGKAN.set_index("Model")
    cms, pred, cek = {}, te[["rs_name", "rating", "text", "y"]].copy(), []
    for nm, clf in M.items():
        pr = Pipeline([("tfidf", feats()), ("clf", clf)]) \
                .fit(tr["x"], tr["y"]).predict(te["x"])
        pred[nm] = pr
        cms[nm] = confusion_matrix(te["y"], pr, labels=LABELS)
        rn = recall_score(te["y"], pr, labels=LABELS, average=None)
        cek.append((nm, round(f1_score(te["y"], pr, average="macro"), 4),
                    LOG.loc[nm, "F1-macro uji"],
                    round(rn[0], 3), LOG.loc[nm, "Recall neg"],
                    round(rn[1], 3), LOG.loc[nm, "Recall netral"]))
        print(f"  {nm:20s} F1 {cek[-1][1]:.4f} (log {cek[-1][2]:.4f}) | "
              f"recall neg {cek[-1][3]} (log {cek[-1][4]}) | "
              f"netral {cek[-1][5]} (log {cek[-1][6]})")

    pred.to_csv(os.path.join(OUT, "prediksi_5model.csv"), index=False,
                encoding="utf-8-sig")
    pd.DataFrame(cek, columns=["Model", "F1 reproduksi", "F1 log",
                               "Rec neg reproduksi", "Rec neg log",
                               "Rec netral reproduksi", "Rec netral log"]) \
        .to_csv(os.path.join(OUT, "cek_reproduksi_vs_log.csv"), index=False)

    rows = []
    for nm, cm in cms.items():
        for i, a in enumerate(LABELS):
            for j, b in enumerate(LABELS):
                rows.append((nm, a, b, int(cm[i, j])))
    pd.DataFrame(rows, columns=["Model", "Label sebenarnya", "Prediksi", "Jumlah"]) \
        .to_csv(os.path.join(OUT, "confusion_matrix_5model.csv"), index=False)

    # SATU FILE PER MODEL
    for nm, cm in cms.items():
        pct = cm / cm.sum(axis=1, keepdims=True)
        ann = np.array([[f"{cm[i, j]}\n({pct[i, j] * 100:.1f}%)"
                         for j in range(3)] for i in range(3)])
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        sns.heatmap(pct, annot=ann, fmt="", cmap="Blues", vmin=0, vmax=1,
                    cbar=True, ax=ax, xticklabels=LABELS, yticklabels=LABELS,
                    linewidths=.5, annot_kws={"fontsize": 11})
        ax.set_title(f"Confusion matrix — {nm}", fontweight="bold")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Sebenarnya")
        simpan(fig, f"cm_{nm.lower().replace(' ', '_')}.png")

    return tr, te, pred


# ------------------------------------------------------------------ 4. EDA
def ringkas(d):
    return pd.Series({
        "Jumlah ulasan": len(d),
        "Teks unik": d["teks_bersih"].str.lower().nunique(),
        "Duplikat (%)": round(100 * d["teks_bersih"].str.lower()
                              .duplicated().mean(), 1),
        "Rata-rata kata": round(d["n_kata"].mean(), 1),
        "Median kata": d["n_kata"].median(),
        "Ulasan <=3 kata (%)": round(100 * (d["n_kata"] <= 3).mean(), 1),
        "Mengandung emoji (%)": round(100 * d["ada_emoji"].mean(), 1),
        "Rata-rata rating": round(d["rating"].mean(), 2),
        "Jumlah RS": d["rs"].nunique(),
        "Berlabel ambigu (%)": round(100 * d["perlu_cek"].mean(), 2),
        "Periode": f"{d['tanggal'].min():%Y-%m} s.d. {d['tanggal'].max():%Y-%m}",
    })


def eda_satu(d, nama, tag):
    """Semua panel jadi file terpisah: eda_<tag>_<panel>.png"""
    # 1) distribusi label
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    c = d["label"].value_counts().reindex(LABELS)
    ax.bar(c.index, c.values, color=[WARNA[k] for k in c.index])
    ax.set_title(f"Distribusi label — {nama}")
    ax.set_ylabel("jumlah ulasan")
    for i, v in enumerate(c.values):
        ax.text(i, v, f"{v}\n({100 * v / c.sum():.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, c.max() * 1.22)
    simpan(fig, f"eda_{tag}_distribusi_label.png")

    # 2) rating per label
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rt = (pd.crosstab(d["label"], d["rating"], normalize="index")
          .reindex(LABELS) * 100)
    rt.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn")
    ax.set_title(f"Komposisi rating per label (%) — {nama}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Rating", bbox_to_anchor=(1.02, 1), loc="upper left",
              ncol=1, fontsize=8)
    simpan(fig, f"eda_{tag}_rating_per_label.png")

    # 3) panjang ulasan per label
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sns.boxplot(data=d, x="label", y="n_kata", order=LABELS, palette=WARNA,
                ax=ax, showfliers=False)
    ax.set_title(f"Panjang ulasan (jumlah kata) per label — {nama}")
    ax.set_xlabel("")
    ax.set_ylabel("jumlah kata")
    simpan(fig, f"eda_{tag}_panjang_ulasan.png")

    # 4) 10 RS terbanyak
    rs = d.groupby("rs").agg(
        n=("label", "size"),
        neg=("label", lambda s: (s == "negatif").mean() * 100),
    ).sort_values("n", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(rs.index[::-1], rs["n"][::-1], color="#5b8db8")
    for y_, (n, ng) in enumerate(zip(rs["n"][::-1], rs["neg"][::-1])):
        ax.text(n, y_, f" {n} ({ng:.0f}% neg)", va="center", fontsize=8)
    ax.set_title(f"10 RS dengan ulasan terbanyak — {nama}")
    ax.set_xlabel("jumlah ulasan")
    ax.set_xlim(0, rs["n"].max() * 1.4)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(9)
    simpan(fig, f"eda_{tag}_top10_rs.png")

    # 5) ulasan per periode
    t = d.dropna(subset=["tanggal"]).copy()
    if len(t):
        per = "Y" if t["tanggal"].dt.year.nunique() > 2 else "M"
        t["periode"] = t["tanggal"].dt.to_period(per).astype(str)
        tt = pd.crosstab(t["periode"], t["label"]).reindex(columns=LABELS)
        fig, ax = plt.subplots(figsize=(10, 4.6))
        tt.plot(ax=ax, marker="o", color=[WARNA[k] for k in LABELS])
        ax.set_title(f"Jumlah ulasan per periode — {nama}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
        simpan(fig, f"eda_{tag}_per_periode.png")

    # 6) kata dominan per label (teks)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")
    ax.set_title(f"Kata dominan per label — {nama}", fontweight="bold")
    cnt = {k: Counter(w for s in d.loc[d["label"] == k, "teks"]
                      for w in kata_bersih(s)).most_common(8)
           for k in LABELS}
    txt = "\n\n".join(f"{k.upper()}: " + ", ".join(f"{w} ({n})" for w, n in v)
                      for k, v in cnt.items())
    ax.text(0.02, 0.98, txt, va="top", fontsize=10, wrap=True)
    simpan(fig, f"eda_{tag}_kata_dominan.png")


def kata_teratas(d, tag, n=12):
    """Satu file per label: kata_teratas_<tag>_<label>.png"""
    for k in LABELS:
        c = Counter(w for s in d.loc[d["label"] == k, "teks"]
                    for w in kata_bersih(s)).most_common(n)[::-1]
        if not c:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.barh([w for w, _ in c], [v for _, v in c], color=WARNA[k])
        ax.set_title(f"{n} kata terbanyak — {k.capitalize()} — {tag} "
                     f"(n={int((d['label'] == k).sum())})")
        ax.set_xlabel("frekuensi")
        for y_, v in enumerate([v for _, v in c]):
            ax.text(v, y_, f" {v}", va="center", fontsize=8)
        ax.set_xlim(0, max(v for _, v in c) * 1.15)
        simpan(fig, f"kata_teratas_{tag.lower().replace(' ', '_')}_{k}.png")


def bagian_eda(d):
    print("[EDA latih & uji]")
    la, uj = d[d["data"] == "Latih"], d[d["data"] == "Uji"]
    eda_satu(la, "Data Latih", "latih")
    eda_satu(uj, "Data Uji", "uji")
    kata_teratas(la, "Data Latih")
    kata_teratas(uj, "Data Uji")

    r = pd.DataFrame({"Data latih": ringkas(la),
                      "Data uji": ringkas(uj),
                      "Gabungan": ringkas(d)})
    r.to_csv(os.path.join(OUT, "eda_statistik_ringkas.csv"))
    print(r.to_string())
    tabel_png(r.reset_index().rename(columns={"index": "Statistik"}),
              "eda_statistik_ringkas.png", "Statistik ringkas dataset",
              lebar=9)

    # Perbandingan latih vs uji -> 4 file terpisah
    # (a) proporsi label
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    p = pd.crosstab(d["data"], d["label"], normalize="index")[LABELS] * 100
    p.plot(kind="bar", stacked=True, ax=ax, color=[WARNA[k] for k in LABELS])
    ax.set_title("Proporsi label: latih vs uji (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    simpan(fig, "eda_banding_proporsi_label.png")

    # (b) sebaran panjang ulasan
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for k, c in {"Latih": "#3b6ea5", "Uji": "#e08e0b"}.items():
        vals = np.log1p(d.loc[d["data"] == k, "n_kata"])
        if len(vals) > 1:
            sns.kdeplot(vals, ax=ax, label=k, color=c, fill=True, alpha=.25)
    ax.set_title("Sebaran panjang ulasan (log1p jumlah kata)")
    ax.set_xlabel("log(1 + jumlah kata)")
    ax.legend(title="Data")
    simpan(fig, "eda_banding_panjang_ulasan.png")

    # (c) proporsi rating
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rr = pd.crosstab(d["data"], d["rating"], normalize="index") * 100
    rr.plot(kind="bar", ax=ax, colormap="RdYlGn")
    ax.set_title("Proporsi rating: latih vs uji (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Rating", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8)
    simpan(fig, "eda_banding_proporsi_rating.png")

    # (d) kesamaan kosakata
    vl = Counter(w for s in la["teks"] for w in kata_bersih(s))
    vu = Counter(w for s in uj["teks"] for w in kata_bersih(s))
    tok_u = sum(vu.values()) or 1
    oov = sum(c for w, c in vu.items() if w not in vl) / tok_u * 100
    jac = len(set(vl) & set(vu)) / len(set(vl) | set(vu)) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axis("off")
    ax.set_title("Kesamaan kosakata latih vs uji", fontweight="bold")
    ax.text(0.02, 0.85,
            f"Kosakata latih : {len(vl):,} kata\n"
            f"Kosakata uji   : {len(vu):,} kata\n"
            f"Irisan (Jaccard): {jac:.1f}%\n"
            f"Token uji yang tidak ada di latih: {oov:.1f}%",
            fontsize=12, va="top", family="monospace")
    simpan(fig, "eda_banding_kosakata.png")


# ------------------------------------------------------------------ 5. wordcloud
def bagian_wordcloud(d):
    print("[EDA gabungan + wordcloud]")
    from wordcloud import WordCloud

    # gabungan -> 3 file terpisah
    # (a) pie label
    fig, ax = plt.subplots(figsize=(6, 5))
    c = d["label"].value_counts().reindex(LABELS)
    ax.pie(c.values,
           labels=[f"{k}\n{v:,} ({100 * v / c.sum():.1f}%)"
                   for k, v in c.items()],
           colors=[WARNA[k] for k in c.index],
           wedgeprops=dict(width=.45), startangle=90)
    ax.set_title(f"Distribusi label (N={len(d):,})")
    simpan(fig, "gabungan_pie_label.png")

    # (b) rating per label
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rr = (pd.crosstab(d["label"], d["rating"], normalize="index")
          .reindex(LABELS) * 100)
    rr.plot(kind="bar", stacked=True, ax=ax, colormap="RdYlGn")
    ax.set_title("Rating per label (%) — gabungan")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Rating", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8)
    simpan(fig, "gabungan_rating_per_label.png")

    # (c) panjang ulasan
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sns.histplot(data=d[d["n_kata"] <= 60], x="n_kata", hue="label",
                 hue_order=LABELS, palette=WARNA, bins=30, element="step",
                 ax=ax)
    ax.set_title("Panjang ulasan (<= 60 kata) — gabungan")
    ax.set_xlabel("jumlah kata")
    simpan(fig, "gabungan_panjang_ulasan.png")

    def wc(freq, cmap):
        return WordCloud(width=1100, height=620, background_color="white",
                         colormap=cmap, max_words=120, prefer_horizontal=.95,
                         random_state=42, collocations=False) \
                .generate_from_frequencies(freq)

    frek = {None: Counter(w for s in d["teks"] for w in kata_bersih(s))}
    for k in LABELS:
        frek[k] = Counter(w for s in d.loc[d["label"] == k, "teks"]
                          for w in kata_bersih(s))

    # wordcloud semua
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.imshow(wc(frek[None], "viridis"))
    ax.axis("off")
    ax.set_title(f"Wordcloud seluruh ulasan (N={len(d):,})", fontweight="bold")
    simpan(fig, "wordcloud_semua.png")

    # wordcloud per label -> 3 file terpisah
    for k, cm in [("positif", "Greens"), ("negatif", "Reds"),
                  ("netral", "Oranges")]:
        fig, ax = plt.subplots(figsize=(11, 6.2))
        ax.imshow(wc(frek[k], cm))
        ax.axis("off")
        ax.set_title(f"Wordcloud — {k.capitalize()} "
                     f"(n={int((d['label'] == k).sum()):,})", fontweight="bold")
        simpan(fig, f"wordcloud_{k}.png")

    # kata khas per label -> 3 file terpisah
    tot = {k: sum(frek[k].values()) for k in LABELS}
    for k in LABELS:
        sk = {}
        for w, n in frek[k].items():
            if n >= 8:
                denom = sum(tot[o] for o in LABELS if o != k) or 1
                lain = sum(frek[o].get(w, 0) for o in LABELS if o != k) / denom
                sk[w] = (n / tot[k]) / (lain + 1e-6) if lain > 0 else (n / tot[k]) / 1e-5
        top = sorted(sk.items(), key=lambda x: -x[1])[:12][::-1]
        if not top:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.barh([w for w, _ in top], [v for _, v in top], color=WARNA[k])
        ax.set_title(f"Kata khas label {k}")
        ax.set_xlabel("rasio frekuensi relatif vs label lain")
        for y_, v in enumerate([v for _, v in top]):
            ax.text(v, y_, f" {v:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, max(v for _, v in top) * 1.15)
        simpan(fig, f"kata_khas_{k}.png")


# ------------------------------------------------------------------ extra
def bagian_extra(d, tr=None, te=None, pred=None):
    print("[EXTRA]")
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import learning_curve, StratifiedKFold

    if tr is None:
        tr, te = muat_latih_uji_model()

    # (a) fitur teratas per kelas -> 3 file terpisah
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95,
                          sublinear_tf=True, max_features=20000)
    Xv = vec.fit_transform(tr["x"])
    lr = LogisticRegression(C=10, max_iter=3000,
                            class_weight="balanced").fit(Xv, tr["y"])
    nama = np.array(vec.get_feature_names_out())
    for i, k in enumerate(lr.classes_):
        idx = np.argsort(lr.coef_[i])[-12:]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.barh(nama[idx], lr.coef_[i][idx], color=WARNA.get(k, "#3b6ea5"))
        ax.set_title(f"Fitur paling mendorong '{k}' "
                     f"(koefisien Logistic Regression)")
        ax.set_xlabel("koefisien")
        for y_, v in enumerate(lr.coef_[i][idx]):
            ax.text(v, y_, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlim(lr.coef_[i][idx].min() * 1.1 if lr.coef_[i][idx].min() < 0 else 0,
                    lr.coef_[i][idx].max() * 1.15)
        simpan(fig, f"extra_fitur_teratas_{k}.png")

    # (b) learning curve -> 2 file terpisah
    for nm, clf in [("SVM linear",
                     SVC(kernel="linear", C=10, class_weight="balanced")),
                    ("Logistic Regression",
                     LogisticRegression(C=10, max_iter=2000,
                                        class_weight="balanced"))]:
        pipe = Pipeline([
            ("t", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                                  sublinear_tf=True, max_features=20000)),
            ("c", clf)])
        sz, a_tr, a_va = learning_curve(
            pipe, tr["x"], tr["y"],
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
            scoring="f1_macro", n_jobs=1)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(sz, a_tr.mean(1), marker="o", label="Latih", color="#3b6ea5")
        ax.fill_between(sz, a_tr.mean(1) - a_tr.std(1),
                        a_tr.mean(1) + a_tr.std(1),
                        alpha=.15, color="#3b6ea5")
        ax.plot(sz, a_va.mean(1), marker="o", label="Validasi (CV)",
                color="#e08e0b")
        ax.fill_between(sz, a_va.mean(1) - a_va.std(1),
                        a_va.mean(1) + a_va.std(1),
                        alpha=.15, color="#e08e0b")
        ax.set_title(f"Learning curve — {nm}")
        ax.set_xlabel("jumlah data latih")
        ax.set_ylabel("F1-macro")
        ax.legend()
        simpan(fig, f"extra_learning_curve_{nm.lower().replace(' ', '_')}.png")

    # (c) kesalahan per RS & panjang -> 2 file terpisah
    if pred is not None:
        p = pred.copy()
        p["salah"] = p["SVM"] != p["y"]
        p["n_kata"] = p["text"].astype(str).str.findall(r"[A-Za-z]+").str.len()
        rs = (p.groupby("rs_name").agg(n=("salah", "size"),
                                       salah=("salah", "mean"))
              .query("n >= 20").sort_values("salah"))

        # (c1) per RS
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(rs.index, rs["salah"] * 100, color="#c0504d")
        ax.set_title("Tingkat kesalahan SVM per RS (n >= 20)")
        ax.set_xlabel("% salah")
        for y_, (nm_, n_, s_) in enumerate(zip(rs.index, rs["n"], rs["salah"])):
            ax.text(s_ * 100, y_, f" {s_ * 100:.1f}% (n={n_})",
                    va="center", fontsize=8)
        ax.set_xlim(0, max(rs["salah"].max() * 100 * 1.35, 1))
        for tick in ax.get_yticklabels():
            tick.set_fontsize(9)
        simpan(fig, "extra_kesalahan_per_rs.png")

        # (c2) per panjang
        p["rentang"] = pd.cut(p["n_kata"], [-1, 3, 8, 20, 1000],
                              labels=["<=3", "4-8", "9-20", ">20"])
        g = p.groupby("rentang").agg(n=("salah", "size"),
                                     salah=("salah", "mean"))
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.bar(g.index.astype(str), g["salah"] * 100, color="#5b8db8")
        ax.set_title("Tingkat kesalahan SVM menurut panjang ulasan (kata)")
        ax.set_ylabel("% salah")
        ax.set_ylim(0, max(g["salah"].max() * 100 * 1.25, 1))
        for i, (n, s) in enumerate(zip(g["n"], g["salah"])):
            ax.text(i, s * 100, f"n={n}", ha="center", va="bottom",
                    fontsize=8)
        simpan(fig, "extra_kesalahan_per_panjang.png")

        sal = p[p["salah"]][["rs_name", "rating", "text", "y", "SVM"]] \
            .rename(columns={"y": "label_sebenarnya", "SVM": "prediksi_SVM"})
        sal.to_csv(os.path.join(OUT, "extra_daftar_kesalahan_SVM.csv"),
                   index=False, encoding="utf-8-sig")
        print(f"  -> daftar {len(sal)} kesalahan SVM disimpan")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    d = baca_data()
    tr = te = pred = None
    if "tabel" in BAGIAN:
        bagian_tabel()
    if "cm" in BAGIAN:
        tr, te, pred = bagian_cm()
    if "eda" in BAGIAN:
        bagian_eda(d)
    if "wordcloud" in BAGIAN:
        bagian_wordcloud(d)
    if "extra" in BAGIAN:
        bagian_extra(d, tr, te, pred)
    print(f"\nSelesai. Semua keluaran ada di folder '{OUT}/'.")