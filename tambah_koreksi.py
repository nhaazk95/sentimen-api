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
    ("Kurang menyenangkan karena pelayanan sangat lamaa", "Negatif"),
    ("Pelayanan kamar operasi sangat baik dan detail", "Positif"),
    ("Sangat baik sekali pelayanan ny", "Positif"),
    ("Pelayanan ny bagus..ramah..", "Positif"),
    ("sakit aja ada parkir 2 rb", "Netral"),
    ("Pelayanan klinik laktasinya kerenn bgtt minnn. Recommended buat ibu baruuuu", "Positif"),
    ("saya kasih bintang", "Netral"),
    ("pelayanan poliklinik rsu medika lestari sangat baik", "Positif"),
    ("Pernah dirawat di situ pelayanan sangat baik", "Positif"),
    ("Bagus pelayanan tidak di ragukann", "Positif"),
    ("Kemarin anakku sakit muntah disertai diare aku niatnya mau rawat jalan tapi ternyata sama dokter anak di suruh rawat inap karena sudah menunjukan gejala dehidrasi ,dan segera masuk IGD...padahal anaku pakai bpjs...tapi penanganannya cepat dan ga ribet sama sekali...kamar pasiennya juga nyaman dan bersih...,terimakasih rsu Medika lestari", "Positif"),
    ("Pengalaman saya oprasi hemoroid di Rs medika lestari Awalnya takut (karena kata orang oprasi hemoroid itu sakit banget dll, )tapi ternyata gak sesakit yang dikatakan orang2. Kalau kalian punya keluhan hemoroid boleh tuh konsul sama dokter bedahnya di sini. Ternyata oprasinya gak sakit, cuma sakit sedikit aja pasca oprasinya. Ditambah lagi semua petugasnya ramah, dokter & dokter spesialisnya ternyata juga ramah bree,perawatnya cekatan juga, petugas kebersihannya juga peduli dengan lingkungan pasien, juru masaknya juga enak masakannya (pas dengan kondisi pasien) Manteplah semakin bagus rs nya. Semoga bisa menolong lebih banyak orang lagi. Dan rezekinya melimpah terus baik karyawanya, Ownernya ataupun pasien.amiin,", "Positif"),
    ("pelayanan menyenangkan", "Positif"),
    ("Layanan satset.. Bisa langsung oeprasi setelah dapat rujukan.. Maantaaapp.. Pasien safety", "Positif"),
    ("mantap", "Positif"),
    ("Suka sekali dengan pelayanan dr. Dwi dokternya ramah, pelayanannya sat set dan ngga sakit sama sekali", "Positif"),
    ("RSU bagus", "Positif"),
    ("Dear elisabeth hospital Im a student who is far from my mom. I have to go to doctor and doing my rutine fisiotherapy twice a week. Thank you for a very good very friendly service, its helping me a lot even i use BPJS. I am very grateful to all the staff who helped me and kindly asked about me.", "Positif"),
    ("pelayanan memuaskan tidak ada kekurangan", "Positif"),
    ("Selama dirumah sakit Ajibarang (ruang bersalin) untuk Bu bidannya ramah-ramah", "Positif"),
    ("WA slow respon bangetttt, Nyediain kontak WA biar bisa daftar online tp balesnya bisa 1 hari kemudian..", "Negatif"),
    ("Pelayanan di rsud Ajibarang memuaskan,saya baru menunggu bapak diruang kenari bawah, terimakasih  atas pelayanan perawat dan dokternya yang baik semoga semakin meningkat pelayanannya", "Positif"),
    ("IGDnya cukup baik", "Positif"),
    ("Kegiatan Pelatihan pencegahan dan pengendalian PPI Angk 1 di RSUD Prof Dr. Margono Soekarjo.... sungguh luar biasa. Ditunggu rekan2 yg mau bergabung", "Positif"),
    ("Farmasi rawat jalan semangat,tetap sabar dan teliti meski pasien tumpuk", "Positif"),
    ("Pelayanan poli nya baik", "Positif"),
    ("Tempat bersih,pelayanan bagus,rekomended", "Positif"),
    ("Baik dengan segi pelayanan dan tindakan", "Positif"),
    ("Excellen", "Positif"),
    ("Pelayanan poli sarafbgs", "Positif"),
    ("Trimakasih atas pelayanan merawat  keluarga  kami,Baik doctor,perawatnya yg ramah.semoga rsu  medical lestari bms,semakin maju kedepanya", "Positif"),
    ("Pelayanan cepat,ramah,,semoga amanah slalu", "Positif"),
    ("Pelayanan cepat tidak di perlambat Ramah", "Positif"),
    ("Pelayanan administrasinya tidak ribet bagi pengguna Bpjs", "Positif"),
    ("Pelayanann kepercayaaan anak anak didiknya saya", "Positif"),
    ("pelayanan kesehatan yang ga pake ribet kalo pake BPJS", "Positif"),
    ("Di RS Medika Lestari buntu nunggunya ngga kelamaan,jadi pulangnya tidak kesiangan", "Positif"),
    ("Selama hamil sllu rutin USG & daftar online via whatsapp awal2 fast respon,tp pas CS pendaftaran nomor WA nya ganti ya ampunnn pelayananya slowrespon sekali !! Balas chatnya plenger bgt ! Udh jelas2 mau daftar online mnyrtakan NIK,nama pasien,poli tujuan,hari & tanggal,malah disuruh lgsg datang ke RS ! Minta nomor antrian online ga dikasih ! Grgr cs plenger aku jd antre dr jam 2 smpe jam 5 sore !! Kecewa bangetttttt", "Negatif"),
    ("Terimakasih untuk pelayanan. Hanya diperbaiki supaya lebih sigap dan cepat tanggap. Jangan jutek klo dipanggil sama pasien.", "Negatif"),
    ("Memuaskan,bagus", "Positif"),
    ("Sangan mudah untuk pelayanan dan kerjasamanya", "Positif"),
    ("Pelayanannya optimal banget RSU ini, rekomen lah", "Positif"),
    ("Sangat baik", "Positif"),
    ("Joosss", "Positif"),
    ("nice", "Positif"),
    ("Kamar mawar 15", "Netral"),
    ("Cepat respon saat pasien butuh bantuan atau cepat bertindak saat di mintai tolong oleh penunggu pasien", "Positif"),
    ("Pelayan ruang perinatologi amah+cepat Tidak mengecewakan", "Positif"),
    ("Perawat nya ramah baik bangeetttt, bersih, kamar anak udah 1kamar 1pasien 👍 tidak mengecewakan", "Positif"),
    ("pelayanananya bagus,amanah", "Positif"),
    ("aku dirawat selama 1 minggu emang sangat baik pelayanannya", "Positif"),
    ("Saya pernah tunggu pasien selama kurang lebih 3 hari,waktu itu cucu saya dirawat ..sakitnya suhu badan tinggi ..susah buang air kecil Pelayanan sangat baik..kamar inapnya nyaman,bersih..lingkungannya juga asri banyak tamannya,tempat ibadahnya juga bagus banget..parkiran aman ,luas,gratis lagi", "Positif"),
    ("Selama anak kami d rwat d ruang kepondang anak lumayan.anak kami merasa terhibur krna bnyak mainan dan dengan perawatnya ox", "Netral"),
    ("Sangat baik pelayanannya", "Positif"),
    ("Untuk kamar bersalin,Alhamdulillah pelayanannya memuaskan,sopan,,,", "Positif"),
    ("Pelayanan kamar bersalinnya memuaskan, dari awal datang sampai mau pulang dilayani dengan baik. Terima kasih buat semuanya 🙏🙏", "Positif"),
    ("Pelayanan di RSUD Ajibarang kamar bersalin mantap👍", "Positif"),
    ("sangat baik", "Positif"),
    ("Overall bagus,saluta pelayanan n keramahanya,untuk perawat thank SDH penuh dedikasi n sabar menghadapi pasien.thank a lot n good job#ruang camar", "Positif"),
    ("Pelayanannya memuasksn", "Positif"),
    ("Pelayanan rumah sakitnya bangus banget, dari masuk rumah sakit sampai pulang", "Positif"),
    ("Sangat baik", "Positif"),
    ("Saya ingin menyampaikan apresiasi kepada RSOP Purwokerto. Selama kurang lebih 5 bulan bolak-balik untuk menjalani pengobatan dan kontrol, saya merasa pelayanan yang diberikan sudah sangat baik, informatif, jelas, dan ramah. Namun, sangat disayangkan pelayanan di bagian informasi pada hari ini kurang memuaskan. Petugas yang berjaga terkesan kurang ramah dan kurang sabar dalam menghadapi keluhan pasien. Menurut saya, bagian informasi adalah salah satu layanan yang paling sering didatangi pasien, terutama pasien lanjut usia yang masih kesulitan menggunakan teknologi. Karena itu, sikap yang lebih ramah dan sabar tentu akan sangat membantu. Saya menuliskan ulasan ini bukan untuk menjatuhkan siapa pun, melainkan sebagai bentuk masukan agar pelayanan ke depannya bisa menjadi lebih baik. Apalagi, hari ini saya mendengar cukup banyak keluhan dari pasien lain setelah saya sendiri menyampaikan keluhan yang saya alami. Terima Kasih 🙏🏻", "Positif")
    ("Istri saya 2x dirawat disini, dan bagi istri yg seorang nakes, dia kagum juga dg nakes tetap tersenyum menghadapi pasien yg banyak smp mlm hari... Dia temukan di RSOP ini... Selamat Ulang Tahun ke 20th mudah2an makin okey dalam melayani pasien.... Konsep tanpa lift nya bagus buat penunggu dan penjenguk pasien utk rajin jalan.... Buat nakesnya juga jalan ya?", "Positif"),
    ("Praktek dokter iman rekomended tuh👍", "Positif"),
    ("dokternya beneran nerangin dengan detail, jawab pertanyaannya jg detail, pasien jadi lebih tau terkait penyakitnya", "Positif")
    ("Tempat nya luas banget. Bagian belakang nyaman banget, banyak spot buat ngadem", "Positif"),
    ("Pelayanan bagus,bersih,memuaskan", "Positif"),
    ("pelayanan sdh bagus,Ramah", "Positif"),
    ("Petugas tidak membeda-bedakan pasien JKn dengan pasien umum", "Positif"),
    ("Bukan hanya pelayanan namun nyaman", "Positif"),
    ("Pelayanan SDH bagus,Tingkatkan LG untuk pekayanan", "Positif"),
    ("Pelanyanan sudah bagys, tolong ditingkatkan lagi", "Positif"),
    ("Pelayanan SDH cukup baik,pertahankan", "Positif"),
    ("pelayanan memuaskan,ruangan bersih,nyaman", "Positif"),
    ("Pelayanan sdh cukup baik, Tingkatkan lagi biar tambah Joos", "Positif"),
    ("Pelayanan untuk specialis mata dan bedah mulut sangat lah ramah", "Positif"),
    ("Love the vibes of this hospital", "Positif"),
    ("tempatnya nya luas, ruangan lengkap, Tempat parkir nya luas", "Positif"),
    ("hebat", "Positif"),
    ("Pasien atas nama tnnyusup pasien dr arinton ruang anyelir 1 pelayanan baik, terus tingkatkan.. terimakasih", "Positif"),
    ("Pasien dengan dokter Anton. Untuk Pelayanan  ramah dan tepat, tidak mengecewakan. Dari kamar tulip 1", "Positif"),
    ("Pasien dokter happy, pasien kamar Dahlia 4.", "Positif"),
    ("Pengalaman saya pada saat mau kontrol nomor surat rujukan terkunci, kemudian saya di arahkan untuk minta nomor rujukan baru ke faskes 1,...dan surat rujukan tersebut tidak bisa langsung di pakai,,nunggu hari berikutnya.sangat menghambat banget...mohon untuk di perbaiki lagi untuk pelayanan khususnya di bagian pendaftaran.", "Negatif"),
    ("Pelayananya sungguh memuaskan,dan bikin betah.", "Positif"),
    ("Pasien Dr Heppy Kamar Dahlia 4 Selama Di rawat di sini perawatnya baik-baik, konsekuensi dalam bertugas,ramah terhadap pasien", "Positif"),
    ("Rekam medis mantul", "Positif"),
    ("Kalo konsultasi ke Sp.kj Kita kira berapa yaa...mohon dijawab", "Netral"),
    ("Nice place..nice support system 🥰", "Positif"),
    ("dokternya enakan , pelayanan ramah pokoknya jangan lupa datang ke rsgpm Unsoed ya klo mau periksa gigi ya", "Positif"),
    ("Cari vaksin untuk anak kesana kemari susah responnya lama,karena saya memastikan dahulu sebelum datang kerumah sakitnya untuk kesediannya, cuman di bunda arif tanya admin langsung di respon cepat, harga vaksin juga masih sangat affordable price. Pertahankan bunda arif untuk pelayanan nya yang fast dan clear", "Positif"),
    ("Benerapa hari yang lalu saya ke rs elisabeth,saran saya mohon adanya unit informasi tersendiri, jadi kalau mau bertanya tidak bingung, masa mau bertanya harus nebeng di unit lain, sedangkan unit - unit yang lainnya sibuk semua, setau saya di rmh sakit manapun begitu masuk lobby sudah langsung ada unit informasi tersendiri, sayang loh rs yg begitu megah tidak punya unit informasi, terimakasih", "Negatif"),
    ("Ketemu perawat yang buat sakit hati cara ngomongnya. Almh ibu sy masuk kondisi kesadaran menurun msh diblg gak bs kooperatif. Gmn mau kooperatif kondisinya kesadarannya jg menurun. Kedua, dy perawat tp sdh langsung bilang ini telat udah susah, maksudnya? Kalau sy sebagai anak pasien kita tau kondisinya memang sdg krg baik si pasien almh ibu sy tp sy memohon utk disampaikan kekeluarga sj jgn di dpn pasien tp si perawat masih blg dokter jg akan blg gt, ini jg gak akan dgr jg sy ngmg. Disitu sy makin naik darah dan langsung marah krn meski gmn pun sy yakin almh tetap bs mendengar hny sj tdk bs merespon. Saat itu yg sy butuhkan adlh semangat baik utk sy ataupun pasien bukan dgn kata2 yg menurut sy tdk pantas sbg seorang perawat. Yg bs menentukan jg seharusnya dokter bukan perawat.", "Negatif"),
    ("rs swasta dengan pelayanan terrr ramahh 🥰", "Positif"),
    ("Ke UGD mau minta surat keterangan sehat, sayang pasien UGD sedang penuh jadi ga bisa dilayani. Tapi petugas penerimanya baik menawarkan tunggu tapi ga tau sampai jam berapa, atau cari ke RS lain dan kasih pilihan RS yang kemungkinan besar bisa. Bersyukur pindah rs mengikutibsaran beliau langsung dapat yang bisa melayani. Makasih kakak❤️❤️", "Positif"),
    ("Layanan bagus,  antrian teratur dan tidak membedakan sy sebagai pasien BPJS", "Positif"),
    ("Kami menjenguk rekan yg sakit diklas 1 dn kagum ..ruangannya seperti vip sangat hnya kurang kulkas saja wajarlah...ad rak sepatu ad tmpat jemuran handuk dn lmari pakaian disain ruangannya jg bagus kombinasi kayu..ad sofa yg besar dn empuk  bsa bt tidur.dn ad ruangan  bt sholat dn jg bt istirahat yg lumayan cukuplah....ad al qur'an dn buku dzukir pagi dan petang...Ma syaa Allooh.....sehingga ketika suami sakit minta dirawat disitu padahal harusnya di rs.lain..", "Positif"),
    ("Secara keseluruhan  RSI Purwokerto sangat baik, dari segi pelayanan, tenaga medis dan sarana prasrananya", "Positif"),
    ("Alhamdulillah, istri lahiran di RSI pelayanannya sangat baik", "Positif"),
    ("Udah beberapa kali ke RSI Purwokerto dan nggak pernah kecewa sama pelayanannya. Mulai dari satpam, bagian pendaftaran, perawat, sampai apotek semuanya informatif dan ramah. Fasilitasnya lengkap, parkiran luas, dan nuansa Islaminya bikin adem.", "Positif"),
    ("Pelayanan lambatt", "Negatif"),
    ("pelayanan UGD sudah 👌", "Netral"),
    ("Ibu saya dirawat d 506 prwat ramh2 ,cekatan", "Positif"),
    ("Pelayanan memuskan", "Positif"),
    ("pelayanan radiologi sangat baik👍👍👍"),
    ("Rawat inap di kamar 508..pelayanan susternya gercep,komunikatif.. dokternya informatif, obat2an manjur..", "Positif")
    ("Pelayanannya udah oke bgt, tempat juga udah nyaman. Cuman saya gasuka orang2 yg diruang operasi kurang ramah, harusnya bikin suasana nyaman dengan posisi pasien yang stress karna mau operasi. Dan mereka berisik bgt, ada yg ngomongin makan lah, bercandalah kaya bukan di RS. Tapi overall udah josjisss bgt sii terutama nurse yg di lantai 4, terbantu bgt. Oiya, masih heran kenapa antri obat itu lama pake bgt ya? Dan saya berterima kasih bgt sama mas mas cleaning service dia terajin sii karna yg lainnya cewek tapi yg paling bersih & telaten si mas mas the only one. Ga sabar pengin hamil & lahiran di hermina!!!!", "Positif")
    ("Pelayan naik", "Positif"),
    ("Utk radiologi Tidak takut buat di USG dr ya ramah bgt", "Positif"),
    ("Perawatan di pantai 4 padma bagus,bersih,rapi", "Positif"),
    ("pelayanan rekam medis RS.HERMINA Ramah & cantik 👍", "Positif"),
    ("pelayanan hd sangat memuaskannnn mas cipto, suster triana, suster indi, suster sofi suster aini, mas dinar dan mas wiwit sangat baik", "Positif"),
    ("ktk pelayanan baik..terimakasih hermina pwt.", "Positif"),
    ("Perawatan di lantai 4 Padma MEMUASKAN", "Positif"),
    ("Kamar no 405 perawat mba rosa Rumah sakit untuk lahiran yang sangat nyaman", "Positif")
    ("Radiologi RS Hermina sangat baik", "Positif"),
    ("dr rumah sdh panik liat keadaan anak, bgtu smpe RS bingung n untung nya ada pak securiti langsung d arahin,, bgtu masuk ruang IGD langsung d tanganin dokter tidak ada drama nunggu atau lambat penanganan,,dokter n staf ramah², ruangan nya nyaman, bersih n sejuk", "Positif")
    ("Dirawat d kamar 512 Pelayanan cukup memuaskan,perawat sopan,dokter sopan,memuaskan", "Positif"),
    ("Alhamdulillah  sangat ouas pelayanan polikliniknya sangat baik", "Positif"),
    ("radiologi sangat baik", "Positif"),
    ("Trimakasih atas pelayanan merawat  keluarga  kami,Baik doctor,perawatnya yg ramah .semoga rsu  medical lestari bms,semakin maju kedepanya", "Positif")
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