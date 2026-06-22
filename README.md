# ☕ Pengelompokan Wilayah Distribusi & Klasifikasi Kualitas Biji Kopi
### Menggunakan K-Means Clustering dan Naive Bayes Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.2-orange?logo=scikit-learn)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green?logo=opencv)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.4-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk:
1. **Mengelompokkan wilayah distribusi** biji kopi berdasarkan karakteristik visual menggunakan algoritma **K-Means Clustering** (unsupervised learning).
2. **Mengklasifikasikan kualitas biji kopi** (Grade A, B, C) dari gambar menggunakan algoritma **Naive Bayes Classification** (supervised learning).

Dataset berupa gambar biji kopi diperoleh dari **Kaggle** dan diolah menggunakan **OpenCV** untuk ekstraksi fitur warna dan tekstur.

---

## 🎯 Tujuan

- Mengekstraksi fitur visual dari gambar biji kopi (warna RGB/HSV, tekstur)
- Mengelompokkan wilayah distribusi kopi menggunakan K-Means
- Mengklasifikasikan kualitas biji kopi menggunakan Naive Bayes
- Mengevaluasi performa model dengan metrik Silhouette Score, Akurasi, dan F1-Score

---

## 🗂️ Struktur Proyek

```
coffee_quality_clustering/
│
├── 📁 dataset/
│   ├── 📁 raw/
│   │   ├── coffee_images/
│   │   │   ├── grade_A/
│   │   │   ├── grade_B/
│   │   │   └── grade_C/
│   │   ├── coffee_quality.csv
│   │   └── distribution_region.csv
│   │
│   └── 📁 processed/
│       ├── features_extracted.csv
│       ├── features_scaled.csv
│       └── labeled_dataset.csv
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_image_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_kmeans_clustering.ipynb
│   ├── 05_naive_bayes_classification.ipynb
│   └── 06_evaluation_visualization.ipynb
│
├── 📁 src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── image_preprocessing.py
│   ├── feature_extraction.py
│   ├── kmeans_model.py
│   ├── naive_bayes_model.py
│   ├── evaluation.py
│   └── visualization.py
│
├── 📁 models/
│   ├── kmeans_model.pkl
│   ├── naive_bayes_model.pkl
│   └── scaler.pkl
│
├── 📁 outputs/
│   ├── 📁 plots/
│   │   ├── elbow_curve.png
│   │   ├── cluster_map.png
│   │   ├── confusion_matrix.png
│   │   ├── feature_distribution.png
│   │   └── image_samples.png
│   └── 📁 reports/
│       ├── clustering_report.txt
│       └── classification_report.txt
│
├── 📁 docs/
│   ├── README.md
│   └── flowchart_sistem.png
│
├── requirements.txt
├── config.py
└── main.py
```

---

## 🔄 Alur Pipeline

```
Dataset Kaggle (CSV + Gambar Biji Kopi)
              ↓
  Preprocessing Gambar (OpenCV)
  [Resize → Grayscale → Noise Removal]
              ↓
  Ekstraksi Fitur
  [Warna RGB/HSV, Tekstur, Histogram]
              ↓
    ┌─────────────────────────┐
    │   K-Means Clustering    │ → Pengelompokan Wilayah Distribusi
    │   (Unsupervised)        │   (Elbow Method untuk nilai K optimal)
    └─────────────────────────┘
              ↓
    Label cluster dijadikan kelas
              ↓
    ┌─────────────────────────┐
    │  Naive Bayes Classifier │ → Klasifikasi Kualitas Biji Kopi
    │  (Supervised)           │   (Grade A / B / C)
    └─────────────────────────┘
              ↓
    Evaluasi & Visualisasi Hasil
    [Silhouette Score, Akurasi, F1-Score, Confusion Matrix]
```

---

## 🛠️ Teknologi yang Digunakan

| Library | Versi | Kegunaan |
|---------|-------|----------|
| Python | 3.10+ | Bahasa pemrograman utama |
| NumPy | 1.26.4 | Komputasi numerik |
| Pandas | 2.2.1 | Manipulasi data CSV |
| Scikit-learn | 1.4.2 | K-Means & Naive Bayes |
| OpenCV | 4.9.0.80 | Preprocessing & ekstraksi fitur gambar |
| Matplotlib | 3.8.4 | Visualisasi grafik |
| Seaborn | 0.13.2 | Visualisasi statistik |
| Pillow | 10.3.0 | Pembacaan gambar |
| Joblib | 1.4.0 | Simpan & load model `.pkl` |
| Jupyter | 1.0.0 | Notebook interaktif |

---

## ⚙️ Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/username/coffee_quality_clustering.git
cd coffee_quality_clustering
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

### 4. Download Dataset dari Kaggle
```bash
pip install kaggle
kaggle datasets download -d schmoyote/coffee-ratings-dataset
# Ekstrak ke folder dataset/raw/
```

---

## 🚀 Cara Menjalankan

### Menjalankan Pipeline Lengkap
```bash
python main.py
```

### Menjalankan per Notebook (Urutan)
```bash
jupyter notebook
```
Buka notebook secara berurutan:
1. `01_data_exploration.ipynb`
2. `02_image_preprocessing.ipynb`
3. `03_feature_extraction.ipynb`
4. `04_kmeans_clustering.ipynb`
5. `05_naive_bayes_classification.ipynb`
6. `06_evaluation_visualization.ipynb`

---

## 📊 Hasil yang Diharapkan

| Metrik | Target |
|--------|--------|
| Silhouette Score (K-Means) | ≥ 0.50 |
| Akurasi (Naive Bayes) | ≥ 80% |
| F1-Score (Naive Bayes) | ≥ 0.75 |

Output visualisasi tersimpan di folder `outputs/plots/`:
- **Elbow Curve** — menentukan jumlah cluster optimal
- **Cluster Map** — peta sebaran wilayah distribusi
- **Confusion Matrix** — evaluasi klasifikasi Naive Bayes
- **Sample Images** — contoh gambar tiap cluster kualitas

---

## 📁 Penjelasan File Utama

| File | Fungsi |
|------|--------|
| `main.py` | Entry point — menjalankan seluruh pipeline |
| `config.py` | Konfigurasi global (path, parameter K, dsb.) |
| `src/data_loader.py` | Membaca CSV & gambar dari folder dataset |
| `src/image_preprocessing.py` | Resize, grayscale, noise removal dengan OpenCV |
| `src/feature_extraction.py` | Ekstraksi fitur warna RGB, HSV, dan tekstur |
| `src/kmeans_model.py` | Training & prediksi model K-Means |
| `src/naive_bayes_model.py` | Training & prediksi model Naive Bayes |
| `src/evaluation.py` | Hitung Silhouette Score, Akurasi, F1-Score |
| `src/visualization.py` | Plot elbow curve, peta wilayah, confusion matrix |

---

## 👤 Penulis

**Nama:** [Nama Anda]  
**NIM:** [NIM Anda]  
**Institusi:** [Nama Universitas/Instansi]  
**Email:** [email@example.com]  

---

## 📄 Lisensi

Proyek ini menggunakan lisensi **MIT**. Lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.

---

> ☕ *"Kopi yang baik dimulai dari biji yang berkualitas — dan data yang baik dimulai dari preprocessing yang tepat."*