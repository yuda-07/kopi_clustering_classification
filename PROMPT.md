
## 🧠 Identitas Proyek

Kamu adalah asisten AI yang membantu mengembangkan proyek machine learning berjudul:

> **"Pengelompokan Wilayah Distribusi dan Klasifikasi Kualitas Biji Kopi  
> Menggunakan K-Means Clustering dan Naive Bayes Classification"**

Proyek ini dikembangkan menggunakan:
- Bahasa: **Python 3.10+**
- Library utama: **Scikit-learn**, **OpenCV**, **Matplotlib**, **Pandas**, **NumPy**
- Dataset: **Gambar biji kopi dari Kaggle** (format `.jpg`/`.png`) + file CSV metadata
- Lingkungan: **Jupyter Notebook** dan script Python modular

---

## 🎯 Tujuan Utama Proyek

Proyek ini memiliki **dua tujuan utama** yang saling berkaitan:

### 1. Pengelompokan Wilayah Distribusi (K-Means Clustering)
- Mengelompokkan data wilayah distribusi kopi berdasarkan **fitur visual** biji kopi
- Menggunakan algoritma **K-Means (Unsupervised Learning)**
- Jumlah cluster `K` ditentukan menggunakan **Elbow Method**
- Output: label cluster untuk setiap data wilayah

### 2. Klasifikasi Kualitas Biji Kopi (Naive Bayes)
- Mengklasifikasikan kualitas biji kopi ke dalam kategori: **Grade A**, **Grade B**, **Grade C**
- Menggunakan algoritma **Gaussian Naive Bayes (Supervised Learning)**
- Input: fitur hasil ekstraksi gambar (warna, tekstur, histogram)
- Label kelas bisa berasal dari: data asli Kaggle atau hasil labeling cluster K-Means

---

## 🖼️ Alur Kerja Sistem (Pipeline)

Berikut alur lengkap yang harus kamu pahami dan ikuti saat membantu proyek ini:

```
[INPUT]
Dataset Kaggle
├── Gambar biji kopi (.jpg/.png) per folder grade
└── File CSV (metadata: asal daerah, skor kualitas, dll.)
        │
        ▼
[TAHAP 1] Preprocessing Gambar (OpenCV)
├── Resize gambar → 128x128 piksel
├── Konversi ke grayscale & RGB/HSV
├── Noise removal (Gaussian Blur)
└── Normalisasi piksel (0–1)
        │
        ▼
[TAHAP 2] Ekstraksi Fitur dari Gambar
├── Rata-rata & standar deviasi channel R, G, B
├── Rata-rata & standar deviasi channel H, S, V
├── Histogram warna (32 bin per channel)
└── Fitur tekstur: kontras, energi, homogenitas (GLCM)
        │
        ▼
[TAHAP 3] K-Means Clustering
├── Input: fitur hasil ekstraksi (features_scaled.csv)
├── Normalisasi fitur dengan StandardScaler
├── Tentukan K optimal → Elbow Method (K=2 s/d 10)
├── Training KMeans(n_clusters=K, random_state=42)
├── Evaluasi: Silhouette Score
└── Output: label cluster per data → labeled_dataset.csv
        │
        ▼
[TAHAP 4] Naive Bayes Classification
├── Input: fitur + label (dari cluster atau label asli)
├── Split data: 80% train, 20% test (random_state=42)
├── Training: GaussianNB()
├── Prediksi & evaluasi: Akurasi, Precision, Recall, F1
└── Output: classification_report.txt + confusion_matrix.png
        │
        ▼
[OUTPUT]
├── Model tersimpan: kmeans_model.pkl, naive_bayes_model.pkl
├── Visualisasi: elbow_curve.png, cluster_map.png, confusion_matrix.png
└── Laporan: clustering_report.txt, classification_report.txt
```

---

## 📂 Struktur Folder yang Harus Diketahui AI

```
coffee_quality_clustering/
├── dataset/
│   ├── raw/
│   │   ├── coffee_images/        ← Gambar input per grade
│   │   ├── coffee_quality.csv    ← Metadata kopi (Kaggle)
│   │   └── distribution_region.csv
│   └── processed/
│       ├── features_extracted.csv
│       ├── features_scaled.csv
│       └── labeled_dataset.csv
├── src/
│   ├── data_loader.py
│   ├── image_preprocessing.py
│   ├── feature_extraction.py
│   ├── kmeans_model.py
│   ├── naive_bayes_model.py
│   ├── evaluation.py
│   └── visualization.py
├── models/
│   ├── kmeans_model.pkl
│   ├── naive_bayes_model.pkl
│   └── scaler.pkl
├── outputs/plots/
├── config.py
└── main.py
```

---

## 📐 Konvensi Kode yang Harus Diikuti

Saat membantu menulis kode, selalu ikuti aturan berikut:

- **Bahasa komentar**: Indonesia
- **Variabel**: snake_case (contoh: `fitur_gambar`, `model_kmeans`)
- **Fungsi**: deskriptif dan modular (1 fungsi = 1 tugas)
- **Random state**: selalu gunakan `random_state=42`
- **Path file**: selalu referensikan dari `config.py`, bukan hardcode
- **Error handling**: selalu gunakan `try-except` untuk operasi file/gambar
- **Logging**: gunakan `print()` sederhana dengan format `[INFO]`, `[ERROR]`, `[OK]`

Contoh format logging:
```python
print("[INFO] Memuat gambar dari dataset...")
print("[OK] Ekstraksi fitur selesai: 500 gambar diproses")
print("[ERROR] File tidak ditemukan: coffee_quality.csv")
```

---

## 🧩 Penjelasan Setiap Modul

### `config.py`
Menyimpan semua konfigurasi global:
- Path dataset, path output, path model
- Parameter K-Means: jumlah cluster, max iterasi
- Parameter preprocessing: ukuran resize, random state

### `data_loader.py`
- Fungsi membaca gambar dari folder per grade
- Fungsi membaca CSV metadata
- Return: list gambar (numpy array) + label

### `image_preprocessing.py`
- Resize gambar ke 128x128
- Konversi BGR → RGB dan BGR → HSV
- Gaussian blur untuk noise removal
- Normalisasi nilai piksel

### `feature_extraction.py`
- Hitung mean & std tiap channel warna
- Hitung histogram warna
- Hitung fitur tekstur GLCM (dari `skimage.feature`)
- Return: DataFrame fitur per gambar

### `kmeans_model.py`
- Normalisasi fitur dengan `StandardScaler`
- Training K-Means dengan Elbow Method
- Simpan model ke `models/kmeans_model.pkl`
- Return: label cluster, nilai inertia, silhouette score

### `naive_bayes_model.py`
- Split data train/test
- Training `GaussianNB`
- Prediksi & evaluasi
- Simpan model ke `models/naive_bayes_model.pkl`

### `evaluation.py`
- Hitung Silhouette Score untuk K-Means
- Hitung Akurasi, Precision, Recall, F1 untuk Naive Bayes
- Cetak laporan ke console & simpan ke file `.txt`

### `visualization.py`
- Plot Elbow Curve (inertia vs K)
- Plot Confusion Matrix (heatmap Seaborn)
- Plot distribusi fitur per cluster
- Plot sampel gambar per grade/cluster

---

## ⚠️ Hal Penting yang Harus Selalu Diingat AI

1. **Dataset berupa gambar** — bukan tabel biasa. Fitur harus diekstraksi dulu dari gambar sebelum bisa dimasukkan ke model.
2. **K-Means bersifat unsupervised** — tidak membutuhkan label saat training. Label baru dihasilkan setelah clustering selesai.
3. **Naive Bayes bersifat supervised** — membutuhkan label kelas. Label bisa berasal dari folder dataset (grade_A/B/C) atau hasil K-Means.
4. **Urutan pipeline harus dijaga** — preprocessing → ekstraksi fitur → clustering → klasifikasi → evaluasi.
5. **Semua path file** harus diambil dari `config.py`, bukan ditulis langsung di dalam fungsi.
6. **Simpan model** setelah training menggunakan `joblib.dump()`.
7. **Evaluasi wajib ditampilkan** setiap kali model dijalankan.

---

## 💬 Cara AI Harus Merespons

Saat diminta membantu proyek ini, kamu harus:

- ✅ Selalu menulis kode yang **lengkap, bisa langsung dijalankan**, bukan potongan
- ✅ Menambahkan **komentar berbahasa Indonesia** di setiap blok kode penting
- ✅ Mengikuti **struktur folder dan nama file** yang sudah ditentukan
- ✅ Menjelaskan **logika di balik kode** secara singkat setelah menulis kode
- ✅ Jika ada pilihan pendekatan, **rekomendasikan yang terbaik** dan jelaskan alasannya
- ❌ Jangan mengubah struktur folder tanpa konfirmasi
- ❌ Jangan hardcode path file langsung di dalam fungsi
- ❌ Jangan menggunakan library selain yang ada di `requirements.txt`

---

## 🏷️ Metadata Proyek

| Keterangan | Detail |
|------------|--------|
| Nama Proyek | Pengelompokan Wilayah Distribusi & Klasifikasi Kualitas Biji Kopi |
| Algoritma 1 | K-Means Clustering (Unsupervised) |
| Algoritma 2 | Gaussian Naive Bayes (Supervised) |
| Input Utama | Gambar biji kopi (.jpg/.png) + CSV metadata |
| Output Utama | Label cluster wilayah + Prediksi grade kualitas |
| Bahasa | Python 3.10+ |
| Library | Scikit-learn, OpenCV, Matplotlib, Pandas, NumPy, Seaborn |
| Dataset | Kaggle (Coffee Bean / Coffee Quality Institute) |
| AI Pembantu | QwenMax |

---

*File ini dibuat sebagai konteks permanen untuk sesi pengembangan proyek bersama AI.*  
*Selalu sertakan file ini di awal sesi chat baru agar AI memahami keseluruhan konteks proyek.*