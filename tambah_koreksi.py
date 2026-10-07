import csv
import pandas as pd
import os

PATH = "data/training/train_pool.csv"

koreksi = [
    ("pelayanan sangat bagus tidak ada kekurangan", "Positif"),
    ("Perfect", "Positif"),
    ("Bagus tempatnya", "Positif"),
    ("baik", "Positif"),
    ("Pelayanan di igd sudah baik", "Positif"),
    ("radiologi baik", "Positif"),
    ("Langganan bunda sudah lama,pasien dokter Arinton ruang melati.pelayanan dokter dan perawat baik.anak cucu sudah langganan rsu bunda lamaa.mks RS bunda", "Positif"),
    ("Sya pasien de sutrisno ibu raniyah sya puasa dg pelyannyaaa", "Positif"),
    ("Tempatnya bagus, pelayanannya engga! Sayang bgt. Sebaiknya management berbenah dan fokus dibidang sdm nya.", "Netral"),
    ("Trimakasih sejak Oktober 2016.", "Netral"),
    ("Pelayanan memuaskan,perawat sopan,dokter sopan", "Positif"),
    ("sudah cukup.", "Netral"),
    ("Tingkatkan lagi pelayanan yg sudah prima", "Netral"),
    ("pelayanan sangat memuaskan", "Positif"),
    ("Andalan keluarga kami dalam berobat dan kontrol kesehatan keluarga kami", "Netral"),
    ("Semoga semakin amanah", "Netral"),
    ("Melayani dengan sepenuh hati", "Positif"),
    ("Bismillah, kemarin saya ada benjolan di PD kiri sejak 2020 awal covid,sempat periksa ke beberapa dokter tapi ragu untuk operasi Awal tahun ini makin besar diniatkan kembali untuk periksa ke dokter Qadarullah dipertemukan dengan dr.davin yang ramah dijelaskan Untuk semuanya bagus, baik dokter sama suster serta pelayanan saat kontrol, tapi Sangat di sayangkan untuk pelayanan pengambilan obat entah kenapa kok sangat TDK ramah dan tdk bisa menjelaskan kendalanya, sangat di sayangkan sekali.", "Netral"),
    ("Nunggu obat lama banget, padahal pakai umum, semua dijadikan satu jadi terakhir yg menerima obat, pembayaran memakai Qris jg tambahan biayanya gede", "Negatif"),
    ("Rs swasta terlengkap di Banyumas dan sekitarnya", "Positif"),
    ("Utk radiologi\nPelayanan baik sekali", "Positif"),
    ("Untuk Pasien Umum Tidak dapat melakukan Pembayaran Non Tunai", "Netral"),
    ("Rumah sakit ini menerima BPJS", "Netral"),
    ("Selama periksa hermina pelayanan ramah sapam daftar dokter penjelasan sopan selalu senyum", "Positif"),
    ("pelayanan rawat jalannya gajelas bgt judes. gada yg ngasih tau buat ambil nomor antriannya sendiri", "Negatif"),
    ("Pasien dokter arinton ruang tulip 12 Pelayanan dokter dan perawat mantap Recommended", "Positif"),
    ("Ka, bpk saya di diagnosa ulkus kornea, kata Dr. Solusinya ditambal karna lapisannya udah tipis. Nah di RSKM bisa ngga ka, selain di RS jogja bandung dan jkt? Mohon infonya ka", "Netral"),
    ("Aku si yess", "Netral"),
    ("Lingkungannya bersih", "Positif"),
    ("masuk gang dikit, dari luar si kelihatannya bersih & terawat", "Netral"),
    ("lbh baik lg klo ada atap diantara 2 gedung IGD/dpn dgn rawat inap/r jalan", "Netral"),
    ("Untuk semuanya bagus, baik dokter sama suster serta pelayanan saat kontrol, tapi Sangat di sayangkan untuk pelayanan pengambilan obat entah kenapa kok sangat TDK ramah dan tdk bisa menjelaskan kendalanya, sangat di sayangkan sekali", "Negatif"),
    ("Pelayanan poli gigi senin-sabtu pukul 13.00-16.00", "Netral"),
    ("ini saya juga lagi cari pengalaman", "Netral"),
    ("Observasi dan pasang pasak gigi di sini. Baru tau ternyata dokter gigi banyak spesialisasinya dan baru menyadari perawatan gigi itu mahal karena biasanya cukup ke dokter gigi umum", "Netral"),
    ("Kasih nomor antrian pengambilan obat di farmasi jiwa.biar ada kepastian sudah sampai nomor berapa","Netral"),
    ("Sediakan tempat tidur untuk yang kontrol","Netral"),
    ("Sudah cukup baik. Mohon dipertahankan", "Positif"),
    ("Pelayanan nya sdh Bagus, tolong ditingkatkan lg y", "Positif"),
    ("Saya kontrol hari ini", "Netral"),
    ("Luas rumah sakitmya", "Netral"),
    ("Menjadi kepercayaan dalam berobat", "Positif"),
    ("mantapp pelayananya", "Positif"),
    ("Mau ke poli kandungan di bantu dari daftar sampai download aplikasi mba petugas administrasi ramah", "Positif"),
    ("ruang ranap arimbi full nyamuk, mungkin dari got depan ranap ya... kalo bisa sih got nya di kasih kapur ajaib supaya jentik2 pd mati", "Negatif"),
    ("pelayanan kesehatan paling kepercayaan untuk pemulihan dan pengobatan keluarga kami", "Positif"),
    ("Bismillah, kemarin saya ada benjolan di PD kiri sejak 2020 awal covid,sempat periksa ke beberapa dokter tapi ragu untuk operasi Awal tahun ini makin besar diniatkan kembali untuk periksa ke dokter Qadarullah dipertemukan dengan dr.davin yang ramah dijelaskan sampai betul2 paham dari yang tadinya takut mau operasi jadi yakin alhamdulilah hasilnya jinak penyembuhan cepat bekas lukanya ngga keliatan setelah operasi masih dibekali pesan untuk jaga pola makan dan olahraga alhamdulilah bisa ditangani oleh dr.davin semoga Allah yang membalas kebaikan dokter mudah2an bisa olahraga rutin seperti dr.davin dan istri", "Positif"),
    ("mba mas dokternya ramahÂ²,playanan jga Oke,sayangnya klo k kantin pas ujan susah.", "Netral"),
    ("maaf kalo bersalin pake bpjs apakah kassa, tisu, kapas bawa sendiri atau dapat dr rs?", "Netral"),
    ("Info biaya dan prosedur tes DNA. Trmksh", "Netral"),
    ("Adakah no lain yg BS kami hubungi? Dari kemarin saya hub no yg tertera di bio, katanya nomor TDK teregister.", "Netral"),
    ("Ada nomer telepon yang bisa komunikasi lewat watsap gak??", "Netral"),
    ("Pelayanan krng, petugas krng bnyk, lambat smua", "Negatif"),
    ("Kurang menyenangkan karena pelayanan sangat lamaa", "Negatif")
]

df = pd.read_csv(PATH)
if "koreksi" not in df.columns:
    df["koreksi"] = 0
df["koreksi"] = df["koreksi"].fillna(0).astype(int)

if os.path.exists("koreksi_dari_dashboard.csv"):
    dash = pd.read_csv("koreksi_dari_dashboard.csv")
    sudah = {t.strip().lower() for t, _ in koreksi}
    for t, l in zip(dash["text"], dash["label"]):
        if t.strip().lower() not in sudah:
            koreksi.append((t, l))
    print(f"Dari dashboard: {len(dash)} baris dibaca")

baru = pd.DataFrame(koreksi, columns=["text", "label"])
baru["koreksi"] = 1

norm = lambda s: s.str.strip().str.lower()
kunci = dict(zip(norm(baru["text"]), baru["label"]))

# 1. Yang sudah ada: set label benar + tandai koreksi
ada = norm(df["text"]).isin(kunci)
df.loc[ada, "label"] = norm(df.loc[ada, "text"]).map(kunci)
df.loc[ada, "koreksi"] = 1
print(f"Ditandai koreksi: {ada.sum()} baris")

# 2. Yang belum ada: tambahkan sebagai baris baru
belum = ~norm(baru["text"]).isin(set(norm(df["text"])))
df = pd.concat([df, baru[belum]], ignore_index=True)
print(f"Ditambahkan baru: {belum.sum()} baris")

TEST_PATH = "data/training/test_set.csv"
test = pd.read_csv(TEST_PATH)

kunci_test = norm(test["text"])
bentrok = baru[norm(baru["text"]).isin(set(kunci_test))]
if len(bentrok):
    print(f"\nPERINGATAN: {len(bentrok)} koreksi juga ada di test_set.csv:")
    for _, r in bentrok.iterrows():
        lama = test.loc[kunci_test == r["text"].strip().lower(), "label"].unique().tolist()
        print(f"  {r['text'][:50]!r} | label koreksi={r['label']} | label di test_set={lama}")

df.to_csv(PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(df["label"].value_counts().to_dict(), "| koreksi=1:", int(df["koreksi"].sum()))