# Pengelompokan Wilayah Distribusi & Klasifikasi Kualitas Biji Kopi
### Menggunakan K-Means Clustering dan Naive Bayes Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.2-orange?logo=scikit-learn)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green?logo=opencv)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.4-red)
![scikit-image](https://img.shields.io/badge/scikit--image-0.22.0-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Deskripsi Proyek

Proyek ini bertujuan untuk:
1. **Mengelompokkan wilayah distribusi** biji kopi berdasarkan karakteristik visual menggunakan algoritma **K-Means Clustering** (unsupervised learning).
2. **Mengklasifikasikan kualitas biji kopi** dari gambar menggunakan algoritma **Naive Bayes Classification** (supervised learning).

Dataset berupa **1.913 gambar biji kopi** dari 3 jenis: **Arabika** (633), **Liberika** (639), dan **Robusta** (641), diolah menggunakan **OpenCV** untuk ekstraksi fitur warna, histogram, dan tekstur GLCM.

---

## Tujuan

- Mengekstraksi 112 fitur visual dari gambar biji kopi (warna RGB/HSV, histogram 32-bin, tekstur GLCM)
- Mengelompokkan wilayah distribusi kopi menggunakan K-Means dengan Elbow Method
- Deteksi dan hapus outlier berdasarkan Z-score sebelum clustering
- Mengklasifikasikan kualitas biji kopi menggunakan Gaussian Naive Bayes
- Mengevaluasi performa model dengan metrik Silhouette Score, Akurasi, Presisi, Recall, dan F1-Score
- Menghasilkan 12+ visualisasi statistik lengkap

---

## Struktur Proyek

```
kopi_clustering_classification/
|
|-- dataset/
|   |-- raw/
|   |   |-- coffee_images/
|   |   |   |-- arabika/         (633 gambar)
|   |   |   |-- liberika/        (639 gambar)
|   |   |   |-- robusta/         (641 gambar)
|   |   |-- coffee_quality.csv
|   |   |-- distribution_region.csv
|   |
|   |-- processed/
|       |-- features_extracted.csv
|       |-- features_scaled.csv
|       |-- labeled_dataset.csv
|
|-- src/
|   |-- __init__.py
|   |-- data_loader.py           # Membaca gambar & CSV metadata
|   |-- image_preprocessing.py   # Resize, blur, konversi warna, normalisasi
|   |-- feature_extraction.py    # RGB/HSV stats, histogram, GLCM tekstur
|   |-- kmeans_model.py          # StandardScaler, Elbow, outlier removal, K-Means
|   |-- naive_bayes_model.py     # Gaussian Naive Bayes + stratified split
|   |-- evaluation.py            # Silhouette, akurasi, confusion matrix, laporan
|   |-- visualization.py         # 13 fungsi plot (popup + simpan PNG)
|
|-- models/
|   |-- kmeans_model.pkl
|   |-- naive_bayes_model.pkl
|   |-- scaler.pkl
|
|-- outputs/
|   |-- plots/                   # 12 visualisasi PNG (lihat di bawah)
|   |-- clustering_report.txt
|   |-- classification_report.txt
|
|-- config.py                    # Konfigurasi global (path, parameter)
|-- main.py                      # Entry point pipeline
|-- requirements.txt
|-- PROMPT.md
```

---

## Alur Pipeline

```
Dataset Gambar Biji Kopi (arabika/liberika/robusta)
              |
  TAHAP 1: Pemuatan Data
  [Baca 1.913 gambar dari 3 folder + CSV metadata]
              |
  TAHAP 2: Preprocessing Gambar (OpenCV)
  [Resize 128x128 -> Gaussian Blur -> RGB/HSV -> Normalisasi 0-1]
              |
  TAHAP 3: Ekstraksi Fitur
  [Mean & Std RGB/HSV + Histogram 32-bin + GLCM Texture]
  = 112 fitur per gambar
              |
  TAHAP 4: K-Means Clustering
  [StandardScaler -> Hapus Outlier (z>3.5) -> Elbow Method (K=2..10)
   -> Balanced K Selection -> Training -> Silhouette Evaluation]
              |
  Label cluster dijadikan kelas untuk klasifikasi
              |
  TAHAP 5: Naive Bayes Classification
  [Split 80/20 -> GaussianNB -> Prediksi -> Evaluasi]
              |
  TAHAP 5E: Visualisasi Statistik Lengkap
  [12 plot statistik: PCA, korelasi, silhouette, radar, dll]
              |
  TAHAP 6: Ringkasan Hasil + Output File
```

---

## Teknologi yang Digunakan

| Library | Versi | Kegunaan |
|---------|-------|----------|
| Python | 3.10+ | Bahasa pemrograman utama |
| NumPy | 1.26.4 | Komputasi numerik & array |
| Pandas | 2.2.1 | Manipulasi data CSV |
| Scikit-learn | 1.4.2 | K-Means, Naive Bayes, StandardScaler, PCA |
| OpenCV | 4.9.0.80 | Preprocessing & ekstraksi fitur gambar |
| scikit-image | 0.22.0 | GLCM texture features |
| Matplotlib | 3.8.4 | Visualisasi grafik & popup display |
| Seaborn | 0.13.2 | Heatmap, boxplot, pairplot |
| Pillow | 10.3.0 | Pembacaan gambar |
| Joblib | 1.4.0 | Simpan & load model `.pkl` |

---

## Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/username/kopi_clustering_classification.git
cd kopi_clustering_classification
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Siapkan Dataset
Pastikan gambar biji kopi tersedia di `dataset/raw/coffee_images/` dengan subfolder:
```
coffee_images/
|-- arabika/     # gambar biji kopi arabika (.jpg/.png)
|-- liberika/    # gambar biji kopi liberika (.jpg/.png)
|-- robusta/     # gambar biji kopi robusta (.jpg/.png)
```

---

## Cara Menjalankan

### Pipeline Lengkap (semua tahap)
```bash
python main.py
```

Pipeline akan menjalankan 7 tahap secara berurutan:
1. **Inisialisasi** — membuat folder yang diperlukan
2. **Pemuatan Data** — membaca 1.913 gambar dari 3 jenis kopi
3. **Preprocessing** — resize, blur, konversi warna, normalisasi
4. **Ekstraksi Fitur** — 112 fitur per gambar
5. **K-Means Clustering** — normalisasi, hapus outlier, Elbow Method, training
6. **Naive Bayes** — split 80/20, training, prediksi, evaluasi
7. **Visualisasi Statistik** — 12 plot statistik lengkap (popup + PNG)

### Menjalankan Modul Terpisah
```bash
python src/data_loader.py          # Test pemuatan data
python src/image_preprocessing.py  # Test preprocessing
python src/feature_extraction.py   # Test ekstraksi fitur
python src/kmeans_model.py         # Test K-Means
python src/naive_bayes_model.py    # Test Naive Bayes
python src/evaluation.py           # Test evaluasi
python src/visualization.py        # Test visualisasi
```

### Mode Visualisasi
```python
# Di config.py:
SHOW_PLOTS = True   # True = popup + simpan PNG
                    # False = hanya simpan PNG (headless/CI)
```

---

## Hasil Visualisasi Statistik

Berikut semua plot yang dihasilkan pipeline, tersimpan di `outputs/plots/`:

### 1. Elbow Curve
Menentukan jumlah cluster (K) optimal berdasarkan penurunan inertia.

![Elbow Curve](outputs/plots/elbow_curve.png)

### 2. Silhouette Score per K
Membandingkan kualitas clustering untuk setiap nilai K.

![Silhouette per K](outputs/plots/silhouette_scores.png)

### 3. Distribusi Fitur per Cluster
Boxplot perbandingan 6 fitur teratas di setiap cluster.

![Feature Distribution](outputs/plots/feature_distribution.png)

### 4. Confusion Matrix
Heatmap evaluasi klasifikasi Naive Bayes — prediksi vs label sebenarnya.

![Confusion Matrix](outputs/plots/confusion_matrix.png)

### 5. PCA Scatter 2D
Visualisasi posisi cluster di ruang 2D menggunakan Principal Component Analysis.

![PCA Scatter 2D](outputs/plots/pca_scatter_2d.png)

### 6. Correlation Heatmap
Korelasi antar 20 fitur dengan varians tertinggi — mengidentifikasi fitur yang redundant.

![Correlation Heatmap](outputs/plots/correlation_heatmap.png)

### 7. Silhouette Analysis (Per-Sampel)
Plot "pisau" silhouette yang menunjukkan seberapa baik setiap data berada di cluster-nya.

![Silhouette Analysis](outputs/plots/silhouette_analysis.png)

### 8. Distribusi RGB per Jenis Kopi
Perbandingan distribusi warna channel R, G, B antara arabika, liberika, dan robusta.

![RGB Distribution](outputs/plots/rgb_distribution_per_type.png)

### 9. Ukuran Cluster (Bar + Pie Chart)
Jumlah dan proporsi data di setiap cluster.

![Cluster Size](outputs/plots/cluster_size_bar.png)

### 10. Radar Chart (Profil Fitur per Cluster)
Spider chart yang menunjukkan profil rata-rata fitur untuk setiap cluster.

![Radar Chart](outputs/plots/radar_chart_cluster.png)

### 11. Pair Plot (Scatter Matrix)
Matriks scatter plot dari 5 fitur teratas, diwarnai per cluster, dengan KDE diagonal.

![Pair Plot](outputs/plots/pair_plot_top_features.png)

### 12. Perbandingan Metrik Evaluasi
Bar chart perbandingan Akurasi, Presisi, Recall, dan F1-Score dari Naive Bayes.

![Metric Comparison](outputs/plots/metric_comparison.png)

---

## Hasil Evaluasi Model

### K-Means Clustering
| Metrik | Nilai |
|--------|-------|
| Jumlah Cluster | 8 |
| Silhouette Score | 0.2363 |
| Inertia | 41.250,30 |

### Naive Bayes Classification
| Metrik | Nilai |
|--------|-------|
| Akurasi | 97.86% |
| Presisi | 0.9788 |
| Recall | 0.9786 |
| F1-Score | 0.9786 |

---

## Penjelasan File Utama

| File | Fungsi |
|------|--------|
| `main.py` | Entry point — menjalankan seluruh pipeline 7 tahap |
| `config.py` | Konfigurasi global (path dataset/model/output, parameter K, random state) |
| `src/data_loader.py` | Membaca gambar dari folder per jenis kopi + CSV metadata |
| `src/image_preprocessing.py` | Resize 128x128, Gaussian Blur, konversi BGR->RGB/HSV, normalisasi |
| `src/feature_extraction.py` | Mean/Std RGB/HSV, histogram 32-bin, GLCM (kontras, energi, homogenitas) |
| `src/kmeans_model.py` | StandardScaler, Z-score outlier removal, Elbow Method, balanced K selection |
| `src/naive_bayes_model.py` | Gaussian Naive Bayes, stratified split dengan fallback |
| `src/evaluation.py` | Silhouette Score, classification report, confusion matrix, laporan .txt |
| `src/visualization.py` | 13 fungsi plot: elbow, PCA, korelasi, radar, pair plot, dll |

---

## Konfigurasi Parameter

Semua parameter bisa diubah di `config.py`:

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `IMG_SIZE` | 128 | Ukuran resize gambar (piksel) |
| `HISTOGRAM_BINS` | 32 | Jumlah bin histogram per channel |
| `K_MIN` / `K_MAX` | 2 / 10 | Rentang K untuk Elbow Method |
| `TEST_SIZE` | 0.2 | Rasio data test (20%) |
| `RANDOM_STATE` | 42 | Seed untuk reproduktibilitas |
| `SHOW_PLOTS` | True | Tampilkan popup plot (False = headless) |
| `GRADE_LABELS` | arabika, liberika, robusta | Nama folder dataset |

---

## Output Files

| File | Deskripsi |
|------|-----------|
| `models/kmeans_model.pkl` | Model K-Means tersimpan |
| `models/naive_bayes_model.pkl` | Model Naive Bayes tersimpan |
| `models/scaler.pkl` | StandardScaler tersimpan |
| `dataset/processed/features_extracted.csv` | 112 fitur mentah per gambar |
| `dataset/processed/features_scaled.csv` | Fitur yang sudah di-standardisasi |
| `dataset/processed/labeled_dataset.csv` | Dataset + label cluster |
| `outputs/plots/*.png` | 12 visualisasi statistik (lihat di atas) |
| `outputs/clustering_report.txt` | Laporan lengkap K-Means |
| `outputs/classification_report.txt` | Laporan lengkap Naive Bayes |

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik/riset.
