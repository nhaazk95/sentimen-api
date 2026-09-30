# Auto-Retraining Pipeline — KlasifikasiSVM

Pipeline retraining otomatis lewat GitHub Actions, dengan satu gerbang manual
(spot-check label) lewat Pull Request.

## Alur singkat

```
data baru masuk (data/incoming/*.csv, kolom "text")
        │
        ▼
[Workflow 1: 01-prepare-data.yml]  <- terjadwal / manual trigger
  - IndoBERT auto-label semua data baru
  - Sampling sebagian -> spot_check_sample.csv
  - Buka branch + Pull Request
        │
        ▼
   KAMU REVIEW MANUAL DI PR
   (isi kolom "reviewed_label" di spot_check_sample.csv)
        │
        ▼
   Merge PR ke branch main
        │
        ▼
[Workflow 2: 02-retrain-deploy.yml]  <- otomatis jalan setelah merge
  - Gabung data hasil review ke train_pool.csv
  - GridSearchCV retrain SVM
  - Evaluasi ke test_set.csv (fixed, tidak pernah berubah)
  - Bandingkan dengan model lama (data/models/current/metrics.json)
      naik/setara -> commit model baru ke data/models/current/  (DEPLOY)
      turun       -> buka GitHub Issue, model lama tetap dipakai
```

## Struktur ini dibuat untuk repo `sentimen-api` kamu (bukan repo baru)

File-file berikut menggantikan/melengkapi yang sudah ada di repo
`sentimen-api`:
- `main.py` — diganti dengan versi yang import preprocessing dari
  `scripts/preprocessing.py` dan baca metrics dari `metrics.json` (bukan
  hardcoded lagi), supaya otomatis update tiap kali model baru di-deploy.
- `scripts/preprocessing.py` — **satu-satunya** sumber logic preprocessing
  (slang dict, stopwords + `kata_penting`, stemming Sastrawi), dipakai baik
  oleh `main.py` (inference) maupun `scripts/train_evaluate.py` (training),
  supaya tidak ada mismatch antara training dan inference.
- `metrics.json` — berisi metrics model yang sedang live. Sudah saya isi
  duluan pakai angka yang ada di `main.py` versi kamu sekarang
  (accuracy 0.9833, dst) sebagai baseline awal.
- `svm_pipeline.pkl`, `label_encoder.pkl`, `slang_dict.json` — **tidak
  diubah**, tetap file kamu yang sekarang.

## Yang perlu kamu siapkan

1. **`data/training/test_set.csv`** — buat sekali di awal, isi label manual
   (bukan dari IndoBERT), dan JANGAN pernah diubah lagi. Ini kunci supaya
   perbandingan antar retrain adil. Kolom: `text,label`.
2. **`data/training/train_pool.csv`** — training set lama kamu saat ini
   (kolom `text,label`). Commit ini duluan sebelum pipeline pertama kali
   jalan.
3. Isi **`data/incoming/`** dengan file CSV berkolom `text` setiap kali ada
   data ulasan baru terkumpul.
4. Sesuaikan `MODEL_NAME` di `scripts/auto_label.py` dengan model IndoBERT
   sentiment yang kamu pakai saat labeling awal (biar konsisten).
5. Kalau hosting API kamu (yang serve `main.py`) tidak auto-redeploy setiap
   ada push ke `main` (misalnya belum ada CD dari GitHub ke Azure/Render),
   tambahkan step trigger redeploy di bagian `# TODO` pada
   `02-retrain-deploy.yml`.
6. **Kalau nanti kamu update `requirements.txt` di repo API** (misalnya naik
   versi scikit-learn), samakan juga versinya di step "Install dependencies"
   pada `.github/workflows/02-retrain-deploy.yml`. Beda versi scikit-learn
   antara saat training vs saat `main.py` load `.pkl` bisa bikin error atau
   hasil prediksi berubah diam-diam.

## Permission yang dibutuhkan

Di **Settings > Actions > General** repo GitHub kamu, aktifkan:
- "Read and write permissions" untuk `GITHUB_TOKEN`
- "Allow GitHub Actions to create and approve pull requests"

## Catatan

- Toleransi performa (`TOLERANCE` di `train_evaluate.py`) diset 0.5% supaya
  model tidak ditolak hanya karena fluktuasi kecil. Sesuaikan sesuai selera.
- `IfError`/fallback pada sisi Copilot Studio (SkorCek) tidak berubah —
  pipeline ini hanya mengganti file `model.pkl` & `vectorizer.pkl` yang
  dipakai backend `PredictSentiment`, jadi tidak perlu ubah topik Copilot
  Studio-nya sama sekali.
