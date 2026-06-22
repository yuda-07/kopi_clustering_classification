"""
main.py - Pipeline Utama Proyek
================================
Mengorchestrasi seluruh alur kerja proyek:
1. Preprocessing Gambar
2. Ekstraksi Fitur
3. K-Means Clustering (Elbow Method + Training)
4. Naive Bayes Classification
5. Evaluasi Model
6. Visualisasi Hasil

Jalankan dengan: python main.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Pastikan path proyek bisa diakses
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import konfigurasi global
import config

# Import semua modul dari paket src
from src import data_loader
from src import image_preprocessing
from src import feature_extraction
from src import kmeans_model
from src import naive_bayes_model
from src import evaluation
from src import visualization


def cetak_banner(teks):
    """Mencetak banner pemisah untuk setiap tahap pipeline."""
    print("\n")
    print("=" * 60)
    print(f"  {teks}")
    print("=" * 60)
    print()


# ============================================================
# TAHAP 0: Inisialisasi & Persiapan
# ============================================================
def tahap_inisialisasi():
    """Membuat semua direktori yang diperlukan dan memverifikasi konfigurasi."""
    cetak_banner("TAHAP 0: INISIALISASI PROYEK")
    config.ensure_directories()
    print(f"[INFO] Path gambar    : {config.COFFEE_IMAGES_DIR}")
    print(f"[INFO] Path model     : {config.MODELS_DIR}")
    print(f"[INFO] Path output    : {config.OUTPUTS_DIR}")
    print(f"[INFO] Random state   : {config.RANDOM_STATE}")


# ============================================================
# TAHAP 1: Pemuatan Data
# ============================================================
def tahap_muat_data():
    """Memuat gambar dari semua folder grade dan CSV metadata."""
    cetak_banner("TAHAP 1: PEMUATAN DATA")

    # Muat gambar dari semua grade
    list_gambar, list_label, list_nama = data_loader.muat_semua_grade()

    if len(list_gambar) == 0:
        print("[ERROR] Tidak ada gambar yang berhasil dimuat!")
        print(f"[INFO] Pastikan gambar biji kopi tersedia di: {config.COFFEE_IMAGES_DIR}")
        print("[INFO] Struktur folder yang diharapkan:")
        for grade in config.GRADE_LABELS:
            print(f"       coffee_images/{grade}/")
        sys.exit(1)

    # Muat CSV metadata (opsional, tidak menghentikan pipeline jika gagal)
    df_kualitas = data_loader.muat_csv_kualitas()
    df_wilayah = data_loader.muat_csv_wilayah()

    return list_gambar, list_label, list_nama, df_kualitas, df_wilayah


# ============================================================
# TAHAP 2: Preprocessing Gambar
# ============================================================
def tahap_preprocessing(list_gambar):
    """Melakukan preprocessing pada semua gambar."""
    cetak_banner("TAHAP 2: PREPROCESSING GAMBAR")
    print("[INFO] Resize ke 128x128, Gaussian Blur, konversi warna, normalisasi...")

    list_hasil_prep = image_preprocessing.preprocessing_batch(list_gambar)

    if len(list_hasil_prep) == 0:
        print("[ERROR] Tidak ada gambar yang berhasil dipreprocessing!")
        sys.exit(1)

    return list_hasil_prep


# ============================================================
# TAHAP 3: Ekstraksi Fitur
# ============================================================
def tahap_ekstraksi_fitur(list_hasil_prep, list_nama):
    """Mengekstrak fitur dari semua gambar yang sudah dipreprocessing."""
    cetak_banner("TAHAP 3: EKSTRAKSI FITUR")

    # Ekstrak fitur dari batch gambar
    df_fitur = feature_extraction.ekstraksi_batch(list_hasil_prep, list_nama)

    # Simpan hasil ekstraksi fitur ke CSV
    feature_extraction.simpan_fitur(df_fitur)

    return df_fitur


# ============================================================
# TAHAP 4: K-Means Clustering
# ============================================================
def tahap_kmeans_clustering(df_fitur):
    """Melakukan clustering menggunakan K-Means dengan Elbow Method."""
    cetak_banner("TAHAP 4: K-MEANS CLUSTERING")

    # 4a. Normalisasi fitur dengan StandardScaler
    print("[SUB-TAHAP 4A] Normalisasi fitur dengan StandardScaler...")
    df_scaled, scaler = kmeans_model.normalisasi_fitur(df_fitur)
    if df_scaled is None:
        print("[ERROR] Normalisasi fitur gagal!")
        sys.exit(1)
    kmeans_model.simpan_scaler(scaler)

    # 4a-2. Hapus outlier berdasarkan Z-score
    print("\n[SUB-TAHAP 4A-2] Deteksi dan hapus outlier...")
    df_scaled = kmeans_model.hapus_outlier(df_scaled, threshold_z=3.5)

    # Simpan fitur yang sudah di-scale
    df_scaled.to_csv(config.FEATURES_SCALED_CSV, index=False)
    print(f"[OK] Fitur scaled disimpan ke: {config.FEATURES_SCALED_CSV}")

    # 4b. Elbow Method untuk menentukan K optimal
    print("\n[SUB-TAHAP 4B] Elbow Method untuk menentukan K optimal...")
    hasil_elbow = kmeans_model.elbow_method(df_scaled)
    k_optimal = hasil_elbow['k_optimal']

    # Update config dengan K optimal
    config.OPTIMAL_K = k_optimal

    # 4c. Training K-Means dengan K optimal
    print(f"\n[SUB-TAHAP 4C] Training K-Means dengan K={k_optimal}...")
    model_km, label_cluster, inertia, sil_score = kmeans_model.training_kmeans(
        df_scaled, n_clusters=k_optimal
    )
    kmeans_model.simpan_model_kmeans(model_km)

    # 4d. Buat dataset berlabel cluster
    print("\n[SUB-TAHAP 4D] Membuat dataset berlabel cluster...")
    df_labeled = kmeans_model.buat_dataset_berlabel(df_scaled, label_cluster)

    # 4e. Evaluasi clustering
    print("\n[SUB-TAHAP 4E] Evaluasi K-Means Clustering...")
    kolom_fitur = [c for c in df_scaled.columns if c != 'nama_file']
    X_fitur = df_scaled[kolom_fitur].values
    hasil_eval_km = evaluation.evaluasi_kmeans(X_fitur, label_cluster, inertia=inertia)
    evaluation.simpan_laporan_clustering(hasil_eval_km)

    # 4f. Visualisasi hasil clustering
    print("\n[SUB-TAHAP 4F] Visualisasi hasil clustering...")
    visualization.plot_elbow_curve(
        hasil_elbow['nilai_k'],
        hasil_elbow['inertia'],
        k_optimal=k_optimal
    )
    visualization.plot_silhouette_per_k(
        hasil_elbow['nilai_k'],
        hasil_elbow['silhouette_scores']
    )
    visualization.plot_distribusi_cluster(df_scaled, label_cluster)

    return df_scaled, df_labeled, scaler, label_cluster, hasil_eval_km, hasil_elbow


# ============================================================
# TAHAP 5: Naive Bayes Classification
# ============================================================
def tahap_naive_bayes(df_labeled, label_encoder_classes=None):
    """Melakukan klasifikasi kualitas menggunakan Naive Bayes."""
    cetak_banner("TAHAP 5: NAIVE BAYES CLASSIFICATION")

    # 5a. Siapkan data training
    print("[SUB-TAHAP 5A] Menyiapkan data training...")
    X, y, nama_fitur, label_encoder = naive_bayes_model.siapkan_data_training(
        df_labeled, kolom_label='cluster_label'
    )
    if X is None:
        print("[ERROR] Gagal menyiapkan data training Naive Bayes!")
        sys.exit(1)

    # 5b. Training model Naive Bayes
    print("\n[SUB-TAHAP 5B] Training Gaussian Naive Bayes...")
    hasil_training = naive_bayes_model.training_naive_bayes(X, y)
    model_nb = hasil_training['model']
    naive_bayes_model.simpan_model_nb(model_nb)

    # 5c. Evaluasi model
    print("\n[SUB-TAHAP 5C] Evaluasi Naive Bayes Classification...")
    hasil_eval_nb = evaluation.evaluasi_naive_bayes(
        hasil_training['y_test'],
        hasil_training['y_pred'],
        label_names=[str(c) for c in label_encoder.classes_]
    )
    evaluation.simpan_laporan_klasifikasi(hasil_eval_nb)

    # 5d. Visualisasi confusion matrix
    print("\n[SUB-TAHAP 5D] Visualisasi Confusion Matrix...")
    visualization.plot_confusion_matrix(
        hasil_training['y_test'],
        hasil_training['y_pred'],
        label_names=[f"Cluster {c}" for c in label_encoder.classes_]
    )

    return hasil_training, hasil_eval_nb, label_encoder


# ============================================================
# TAHAP 6: Ringkasan & Output
# ============================================================
def tahap_ringkasan(hasil_eval_km, hasil_eval_nb):
    """Menampilkan ringkasan akhir seluruh pipeline."""
    cetak_banner("TAHAP 6: RINGKASAN HASIL")

    print("  >> RINGKASAN HASIL PIPELINE")
    print("  " + "-" * 50)

    # Ringkasan K-Means
    print("\n  [K-Means Clustering]")
    print(f"  Jumlah Cluster      : {hasil_eval_km['jumlah_cluster']}")
    print(f"  Silhouette Score    : {hasil_eval_km['silhouette_score']:.4f}")
    print(f"  Inertia             : {hasil_eval_km['inertia']}")
    for cluster, jumlah in hasil_eval_km['distribusi'].items():
        print(f"    Cluster {cluster}      : {jumlah} data")

    # Ringkasan Naive Bayes
    print("\n  [Naive Bayes Classification]")
    print(f"  Akurasi             : {hasil_eval_nb['akurasi']:.4f} ({hasil_eval_nb['akurasi']*100:.2f}%)")
    print(f"  Presisi             : {hasil_eval_nb['presisi']:.4f}")
    print(f"  Recall              : {hasil_eval_nb['recall']:.4f}")
    print(f"  F1-Score            : {hasil_eval_nb['f1_score']:.4f}")

    # Daftar file output
    print("\n  [File Output yang Dihasilkan]")
    print("  " + "-" * 50)
    file_output = [
        ("Model K-Means", config.KMEANS_MODEL_PATH),
        ("Model Naive Bayes", config.NAIVE_BAYES_MODEL_PATH),
        ("Scaler", config.SCALER_PATH),
        ("Fitur Extracted", config.FEATURES_EXTRACTED_CSV),
        ("Fitur Scaled", config.FEATURES_SCALED_CSV),
        ("Labeled Dataset", config.LABELED_DATASET_CSV),
        ("Elbow Curve", config.ELBOW_CURVE_PATH),
        ("Silhouette per K", config.SILHOUETTE_PER_K_PATH),
        ("Feature Distribution", config.FEATURE_DISTRIBUTION_PATH),
        ("Confusion Matrix", config.CONFUSION_MATRIX_PATH),
        ("PCA Scatter 2D", config.PCA_SCATTER_PATH),
        ("Correlation Heatmap", config.CORRELATION_HEATMAP_PATH),
        ("Silhouette Analysis", config.SILHOUETTE_ANALYSIS_PATH),
        ("RGB Distribution", config.RGB_DISTRIBUTION_PATH),
        ("Cluster Size Chart", config.CLUSTER_SIZE_PATH),
        ("Radar Chart", config.RADAR_CHART_PATH),
        ("Pair Plot", config.PAIR_PLOT_PATH),
        ("Metric Comparison", config.METRIC_COMPARISON_PATH),
        ("Laporan Clustering", config.CLUSTERING_REPORT_PATH),
        ("Laporan Klasifikasi", config.CLASSIFICATION_REPORT_PATH),
    ]
    for nama, path in file_output:
        status = "[OK]" if os.path.exists(path) else "[--]"
        print(f"    {status} {nama}: {os.path.basename(path)}")

    print("\n" + "=" * 60)
    print("  PIPELINE SELESAI BERHASIL DIJALANKAN!")
    print("=" * 60)


# ============================================================
# FUNGSI UTAMA: Menjalankan Seluruh Pipeline
# ============================================================
def main():
    """Fungsi utama yang menjalankan seluruh pipeline proyek secara berurutan."""

    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 58 + "|")
    print("|" + "  PENGELOMPOKAN WILAYAH DISTRIBUSI & KLASIFIKASI KUALITAS".center(58) + "|")
    print("|" + "  BIJI KOPI MENGGUNAKAN K-MEANS & NAIVE BAYES".center(58) + "|")
    print("|" + " " * 58 + "|")
    print("+" + "=" * 58 + "+")

    # TAHAP 0: Inisialisasi
    tahap_inisialisasi()

    # TAHAP 1: Pemuatan Data
    list_gambar, list_label, list_nama, df_kualitas, df_wilayah = tahap_muat_data()

    # TAHAP 2: Preprocessing Gambar
    list_hasil_prep = tahap_preprocessing(list_gambar)

    # TAHAP 3: Ekstraksi Fitur
    df_fitur = tahap_ekstraksi_fitur(list_hasil_prep, list_nama)

    # TAHAP 4: K-Means Clustering
    df_scaled, df_labeled, scaler, label_cluster, hasil_eval_km, hasil_elbow = tahap_kmeans_clustering(df_fitur)

    # TAHAP 5: Naive Bayes Classification
    hasil_training_nb, hasil_eval_nb, label_encoder = tahap_naive_bayes(df_labeled)

    # TAHAP 5E: Visualisasi Statistik Lengkap (BARU)
    cetak_banner("TAHAP 5E: VISUALISASI STATISTIK LENGKAP")
    kolom_fitur_nb = [c for c in df_scaled.columns if c != 'nama_file']
    X_fitur_final = df_scaled[kolom_fitur_nb].values

    visualization.jalankan_semua_visualisasi_statistik(
        df_scaled=df_scaled,
        label_cluster=label_cluster,
        X_fitur=X_fitur_final,
        hasil_elbow=hasil_elbow,
        hasil_eval_nb=hasil_eval_nb,
        list_hasil_prep=list_hasil_prep,
        list_label_asli=list_label
    )

    # TAHAP 6: Ringkasan
    tahap_ringkasan(hasil_eval_km, hasil_eval_nb)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
