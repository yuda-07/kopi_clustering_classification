# Pengelompokan Wilayah Distribusi & Klasifikasi Kualitas Biji Kopi
### Menggunakan K-Means Clustering dan Naive Bayes Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.2-orange?logo=scikit-learn)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green?logo=opencv)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.4-red)
![scikit-image](https://img.shields.io/badge/scikit--image-0.22.0-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Deskripsi Proyek

Proyek ini adalah pipeline pembelajaran mesin (*machine learning*) lengkap untuk analisis dan penentuan kualitas biji kopi melalui pemrosesan citra digital (*digital image processing*). Proyek ini menggabungkan dua metode utama:
1. **Unsupervised Learning (K-Means Clustering)**: Mengelompokkan wilayah distribusi biji kopi berdasarkan kesamaan fitur visual seperti warna (RGB/HSV), histogram, dan tekstur GLCM (*Grey-Level Co-occurrence Matrix*).
2. **Supervised Learning (Naive Bayes Classification)**: Mengklasifikasikan kualitas biji kopi berdasarkan label kelompok/wilayah distribusi yang dihasilkan oleh algoritma clustering.

Dataset yang digunakan terdiri dari **1.913 gambar biji kopi** yang terbagi ke dalam 3 varietas: **Arabika** (633), **Liberika** (639), dan **Robusta** (641).

---

## 🚀 Fitur Utama & Pembaruan Terkini

* **Segmentasi Citra & Auto-Cropping Presisi**: Memotong (*cropping*) latar belakang putih studio secara otomatis untuk mengisolasi biji kopi di tengah gambar. Menggunakan deteksi background dinamis dari 4 sudut gambar untuk toleransi variasi pencahayaan, serta *fallback* pemotongan area tengah 22% secara presisi untuk menjamin tingkat perbesaran biji kopi **sama rata (konsisten)** di seluruh visualisasi.
* **Visualisasi Bar Chart dengan Tumpukan Gambar Biji Kopi Asli**: Mengubah representasi batang grafik (*bar chart*) konvensional menjadi tumpukan (*stack*) gambar biji kopi asli tanpa distorsi aspek rasio (*no stretching*). Sumbu Y merepresentasikan jumlah data melalui jumlah tumpukan biji kopi.
* **Ekstraksi Multi-Fitur Lengkap (112 Dimensi)**:
  * Statistik Warna (Mean & Standard Deviation pada ruang warna RGB dan HSV).
  * Histogram Warna (32-bin per channel).
  * Tekstur GLCM (Kontras, Energi, Homogenitas, Dissimilarity).
* **Data Cleaning & Outlier Removal**: Pembersihan data pencilan menggunakan metode Z-score (threshold $Z > 3.5$) untuk memastikan akurasi pengelompokan K-Means.
* **12+ Visualisasi Statistik**: Mulai dari kurva Elbow, analisis Silhouette per sampel, PCA scatter plot 2D, radar chart profil fitur, hingga heatmap korelasi.

---

## 📁 Struktur Proyek

```
kopi_clustering_classification/
│
├── dataset/
│   ├── raw/
│   │   ├── coffee_images/
│   │   │   ├── arabika/         (633 gambar)
│   │   │   ├── liberika/        (639 gambar)
│   │   │   └── robusta/         (641 gambar)
│   │   ├── coffee_quality.csv
│   │   └── distribution_region.csv
│   │
│   └── processed/
│       ├── features_extracted.csv
│       ├── features_scaled.csv
│       └── labeled_dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Membaca gambar & CSV metadata
│   ├── image_preprocessing.py   # Resize, blur, konversi warna, normalisasi
│   ├── feature_extraction.py    # Statistik warna, histogram, GLCM
│   ├── kmeans_model.py          # Scaling, Z-score outlier removal, K-Means
│   ├── naive_bayes_model.py     # Gaussian Naive Bayes + Stratified K-Fold
│   ├── evaluation.py            # Silhouette, akurasi, confusion matrix
│   └── visualization.py         # Visualisasi plot (tumpukan biji kopi, dll)
│
├── models/
│   ├── kmeans_model.pkl
│   ├── naive_bayes_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── plots/                   # 12+ Visualisasi PNG hasil training
│   ├── clustering_report.txt    # Laporan metrik K-Means
│   └── classification_report.txt# Laporan metrik Naive Bayes
│
├── config.py                    # Parameter & path global proyek
├── main.py                      # Script utama untuk menjalankan pipeline
├── test_warna.py                # Script uji coba ekstraksi warna & cropping
└── requirements.txt             # Dependensi pustaka Python
```

---

## ⚙️ Alur Pipeline Sistem

```
Dataset Gambar Biji Kopi (Arabika / Liberika / Robusta)
               │
   TAHAP 1: Pemuatan Data
   [Membaca 1.913 gambar + CSV metadata]
               │
   TAHAP 2: Preprocessing Gambar (OpenCV)
   [Resize 128x128 -> Gaussian Blur -> RGB & HSV Conversion -> Normalisasi]
               │
   TAHAP 3: Ekstraksi Fitur Citra
   [Mean/Std RGB & HSV + Histogram 32-bin + GLCM Tekstur = 112 Fitur]
               │
   TAHAP 4: K-Means Clustering & Data Cleaning
   [StandardScaler -> Outlier Removal (Z > 3.5) -> Elbow Method -> K-Means (K=8)]
               │
   Label cluster (0-7) disimpan ke dataset sebagai target kelas kualitas
               │
   TAHAP 5: Naive Bayes Classification
   [Split Data 80/20 -> Gaussian Naive Bayes -> Prediksi Kualitas -> Evaluasi]
               │
   TAHAP 6: Visualisasi Statistik & Pelaporan
   [12 plot statistik + Laporan kinerja model ke file .txt]
```

---

## 🛠️ Instalasi & Persiapan

### 1. Clone Repository
```bash
git clone https://github.com/username/kopi_clustering_classification.git
cd kopi_clustering_classification
```

### 2. Buat & Aktifkan Virtual Environment
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Letakkan Dataset
Pastikan gambar biji kopi diletakkan sesuai struktur folder berikut:
`dataset/raw/coffee_images/[arabika|liberika|robusta]/*.jpg`

---

## 🎮 Cara Menjalankan

### Menjalankan Seluruh Pipeline
Jalankan entry point utama untuk memproses data dari awal hingga akhir, menyimpan model, dan memperbarui seluruh visualisasi:
```bash
python main.py
```

### Mode Headless (Tanpa Tampilan Popup)
Jika Anda ingin menjalankan script di server tanpa menampilkan window popup GUI dari Matplotlib, ubah nilai berikut di `config.py`:
```python
SHOW_PLOTS = False  # Hanya menyimpan gambar ke folder outputs/plots/
```

---

## 📊 Hasil Visualisasi Utama (`outputs/plots/`)

### 1. Ukuran Cluster & Distribusi Wilayah (`cluster_size_bar.png`)
Menampilkan jumlah data di setiap cluster dalam bentuk **tumpukan biji kopi asli** yang dipotong seragam (*sama rata*). Visualisasi ini juga menampilkan proporsi persentase di sebelah kanan.

![Cluster Size](outputs/plots/cluster_size_bar.png)

### 2. Elbow Curve & Silhouette Score per K
Digunakan untuk menentukan jumlah cluster (K) optimal. Nilai K terpilih adalah **K = 8** karena memberikan penurunan inersia (*Elbow*) yang signifikan dan nilai rata-rata Silhouette Score yang seimbang.

![Elbow Curve](outputs/plots/elbow_curve.png)
![Silhouette Scores](outputs/plots/silhouette_scores.png)

### 3. PCA Scatter Plot 2D (`pca_scatter_2d.png`)
Reduksi dimensi fitur visual ke dalam 2 komponen utama (PC1 & PC2) untuk memvisualisasikan persebaran data di 8 wilayah distribusi cluster secara spasial.

![PCA 2D](outputs/plots/pca_scatter_2d.png)

### 4. Radar Chart Profil Fitur per Cluster (`radar_chart_cluster.png`)
Menunjukkan karakteristik visual (warna dan tekstur) unik yang mendefinisikan masing-masing dari 8 cluster yang terbentuk.

![Radar Chart](outputs/plots/radar_chart_cluster.png)

---

## 📈 Kinerja & Hasil Evaluasi Model

### K-Means Clustering (Pengelompokan Wilayah)
* **Jumlah Wilayah Optimal (K)**: 8 Cluster
* **Silhouette Score**: 0.2363
* **Inertia**: 41,250.30

### Naive Bayes Classification (Klasifikasi Kualitas)
* **Akurasi**: 97.86%
* **Presisi**: 97.88%
* **Recall (Sensitivitas)**: 97.86%
* **F1-Score**: 97.86%

*Klasifikasi ini membuktikan bahwa label pengelompokan kualitas yang dibentuk secara unsupervised oleh K-Means memiliki pola visual yang sangat konsisten, sehingga Gaussian Naive Bayes mampu mempelajari batas keputusan dengan tingkat akurasi mencapai ~98%.*

---

## ⚖️ Lisensi
Proyek ini dilisensikan di bawah MIT License - bebas digunakan untuk kepentingan akademis, riset, dan pengembangan sistem klasifikasi komoditas pangan.
