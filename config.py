"""
config.py - Konfigurasi Global Proyek
======================================
Menyimpan semua parameter, path file, dan konstanta yang digunakan
di seluruh pipeline proyek ini.
"""

import os

# ============================================================
# PATH DASAR PROYEK
# ============================================================
# Ambil path root proyek secara otomatis berdasarkan lokasi config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PATH DATASET
# ============================================================
# Folder dataset mentah (gambar biji kopi + CSV metadata)
RAW_DATA_DIR = os.path.join(BASE_DIR, "dataset", "raw")
COFFEE_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "coffee_images")
COFFEE_QUALITY_CSV = os.path.join(RAW_DATA_DIR, "coffee_quality.csv")
DISTRIBUTION_REGION_CSV = os.path.join(RAW_DATA_DIR, "distribution_region.csv")

# Folder dataset hasil olahan (fitur & label)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed")
FEATURES_EXTRACTED_CSV = os.path.join(PROCESSED_DATA_DIR, "features_extracted.csv")
FEATURES_SCALED_CSV = os.path.join(PROCESSED_DATA_DIR, "features_scaled.csv")
LABELED_DATASET_CSV = os.path.join(PROCESSED_DATA_DIR, "labeled_dataset.csv")

# ============================================================
# PATH MODEL
# ============================================================
# Folder penyimpanan model yang sudah ditraining
MODELS_DIR = os.path.join(BASE_DIR, "models")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
NAIVE_BAYES_MODEL_PATH = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# ============================================================
# PATH OUTPUT & VISUALISASI
# ============================================================
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

# File output visualisasi - plot utama
ELBOW_CURVE_PATH = os.path.join(PLOTS_DIR, "elbow_curve.png")
CLUSTER_MAP_PATH = os.path.join(PLOTS_DIR, "cluster_map.png")
CONFUSION_MATRIX_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
FEATURE_DISTRIBUTION_PATH = os.path.join(PLOTS_DIR, "feature_distribution.png")
SAMPLE_IMAGES_PATH = os.path.join(PLOTS_DIR, "sample_images.png")
SILHOUETTE_PER_K_PATH = os.path.join(PLOTS_DIR, "silhouette_scores.png")

# File output visualisasi - plot statistik tambahan (BARU)
PCA_SCATTER_PATH = os.path.join(PLOTS_DIR, "pca_scatter_2d.png")
CORRELATION_HEATMAP_PATH = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
SILHOUETTE_ANALYSIS_PATH = os.path.join(PLOTS_DIR, "silhouette_analysis.png")
RGB_DISTRIBUTION_PATH = os.path.join(PLOTS_DIR, "rgb_distribution_per_type.png")
CLUSTER_SIZE_PATH = os.path.join(PLOTS_DIR, "cluster_size_bar.png")
RADAR_CHART_PATH = os.path.join(PLOTS_DIR, "radar_chart_cluster.png")
PAIR_PLOT_PATH = os.path.join(PLOTS_DIR, "pair_plot_top_features.png")
METRIC_COMPARISON_PATH = os.path.join(PLOTS_DIR, "metric_comparison.png")

# Konfigurasi tampilan visualisasi
SHOW_PLOTS = True  # True = tampilkan popup, False = hanya simpan PNG

# File output laporan
CLUSTERING_REPORT_PATH = os.path.join(OUTPUTS_DIR, "clustering_report.txt")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "classification_report.txt")

# ============================================================
# PARAMETER PREPROCESSING GAMBAR
# ============================================================
# Ukuran resize gambar (piksel)
IMG_SIZE = 128

# Kernel Gaussian Blur untuk noise removal
BLUR_KERNEL_SIZE = (5, 5)

# ============================================================
# PARAMETER EKSTRAKSI FITUR
# ============================================================
# Jumlah bin histogram per channel warna
HISTOGRAM_BINS = 32

# ============================================================
# PARAMETER K-MEANS CLUSTERING
# ============================================================
# Rentang nilai K untuk Elbow Method (K=2 sampai K_MAX)
K_MIN = 2
K_MAX = 10

# Jumlah iterasi maksimum K-Means
KMEANS_MAX_ITER = 300

# Jumlah inisialisasi K-Means (n_init)
KMEANS_N_INIT = 10

# Jumlah cluster optimal (akan di-override oleh Elbow Method jika None)
OPTIMAL_K = None

# ============================================================
# PARAMETER NAIVE BAYES CLASSIFICATION
# ============================================================
# Rasio pembagian data train/test
TEST_SIZE = 0.2

# ============================================================
# PARAMETER UMUM
# ============================================================
# Random state untuk semua operasi acak (reproduktibilitas)
RANDOM_STATE = 42

# Daftar jenis biji kopi (label kelas berdasarkan folder dataset)
GRADE_LABELS = ["arabika", "liberika", "robusta"]

# Format file gambar yang didukung
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ============================================================
# FUNGSI BANTUAN: Pastikan semua folder penting sudah ada
# ============================================================
def ensure_directories():
    """
    Membuat semua folder yang diperlukan jika belum ada.
    Dipanggil di awal pipeline untuk menghindari error path not found.
    """
    dirs_to_create = [
        RAW_DATA_DIR,
        COFFEE_IMAGES_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        PLOTS_DIR,
    ]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    print("[OK] Semua direktori proyek sudah tersedia.")


# Jalankan otomatis saat config di-import
if __name__ == "__main__":
    ensure_directories()
    print("[INFO] Konfigurasi proyek berhasil dimuat.")
    print(f"[INFO] Base Dir : {BASE_DIR}")
    print(f"[INFO] Images  : {COFFEE_IMAGES_DIR}")
    print(f"[INFO] Models  : {MODELS_DIR}")
    print(f"[INFO] Outputs : {OUTPUTS_DIR}")
