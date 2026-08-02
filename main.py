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
# STAGE 0: Initialization & Setup
# ============================================================
def tahap_inisialisasi():
    """Create all required directories and verify configuration."""
    cetak_banner("STAGE 0: PROJECT INITIALIZATION")
    config.ensure_directories()
    print(f"[INFO] Images path   : {config.COFFEE_IMAGES_DIR}")
    print(f"[INFO] Models path   : {config.MODELS_DIR}")
    print(f"[INFO] Outputs path  : {config.OUTPUTS_DIR}")
    print(f"[INFO] Random state  : {config.RANDOM_STATE}")


# ============================================================
# STAGE 1: Data Loading
# ============================================================
def tahap_muat_data():
    """Load images from all grade folders and metadata CSVs."""
    cetak_banner("STAGE 1: DATA LOADING")

    # Load images from all grades
    list_gambar, list_label, list_nama = data_loader.muat_semua_grade()

    if len(list_gambar) == 0:
        print("[ERROR] No images loaded successfully!")
        print(f"[INFO] Make sure coffee bean images exist in: {config.COFFEE_IMAGES_DIR}")
        print("[INFO] Expected directory structure:")
        for grade in config.GRADE_LABELS:
            print(f"       coffee_images/{grade}/")
        sys.exit(1)

    # Load metadata CSVs
    df_kualitas = data_loader.muat_csv_kualitas()
    df_wilayah = data_loader.muat_csv_wilayah()

    return list_gambar, list_label, list_nama, df_kualitas, df_wilayah


# ============================================================
# STAGE 2: Image Preprocessing
# ============================================================
def tahap_preprocessing(list_gambar):
    """Preprocess all images."""
    cetak_banner("STAGE 2: IMAGE PREPROCESSING")
    print("[INFO] Resizing to 128x128, Gaussian Blur, color conversion, normalization...")

    list_hasil_prep = image_preprocessing.preprocessing_batch(list_gambar)

    if len(list_hasil_prep) == 0:
        print("[ERROR] No images were preprocessed successfully!")
        sys.exit(1)

    return list_hasil_prep


# ============================================================
# STAGE 3: Feature Extraction
# ============================================================
def tahap_ekstraksi_fitur(list_hasil_prep, list_nama, list_label=None):
    """Extract features from all preprocessed images."""
    cetak_banner("STAGE 3: FEATURE EXTRACTION")

    df_fitur = feature_extraction.ekstraksi_batch(list_hasil_prep, list_nama, list_label=list_label)
    feature_extraction.simpan_fitur(df_fitur)

    return df_fitur


# ============================================================
# STAGE 4: K-Means Clustering
# ============================================================
def tahap_kmeans_clustering(df_fitur):
    """Perform K-Means clustering with Elbow Method."""
    cetak_banner("STAGE 4: K-MEANS CLUSTERING")

    # 4a. Feature normalization with StandardScaler
    print("[SUB-STAGE 4A] Normalizing features with StandardScaler...")
    df_scaled, scaler = kmeans_model.normalisasi_fitur(df_fitur)
    if df_scaled is None:
        print("[ERROR] Feature normalization failed!")
        sys.exit(1)
    kmeans_model.simpan_scaler(scaler)

    # 4a-2. Remove outliers based on Z-score
    print("\n[SUB-STAGE 4A-2] Detecting and removing outliers...")
    df_scaled = kmeans_model.hapus_outlier(df_scaled, threshold_z=5.0)

    # Save scaled features
    df_scaled.to_csv(config.FEATURES_SCALED_CSV, index=False)
    print(f"[OK] Scaled features saved to: {config.FEATURES_SCALED_CSV}")

    # 4b. Elbow Method to determine optimal K
    print("\n[SUB-STAGE 4B] Elbow Method to determine optimal K...")
    hasil_elbow = kmeans_model.elbow_method(df_scaled)
    k_optimal = hasil_elbow['k_optimal']

    # Update config with optimal K
    config.OPTIMAL_K = k_optimal

    # 4c. K-Means training with optimal K
    print(f"\n[SUB-STAGE 4C] Training K-Means with K={k_optimal}...")
    model_km, label_cluster, inertia, sil_score = kmeans_model.training_kmeans(
        df_scaled, n_clusters=k_optimal
    )
    kmeans_model.simpan_model_kmeans(model_km)

    # 4d. Create cluster-labeled dataset
    print("\n[SUB-STAGE 4D] Creating cluster-labeled dataset...")
    df_labeled = kmeans_model.buat_dataset_berlabel(df_scaled, label_cluster)

    # 4e. Evaluate clustering
    print("\n[SUB-STAGE 4E] Evaluating K-Means Clustering...")
    kolom_fitur = [c for c in df_scaled.columns if c not in ['nama_file', 'jenis_kopi']]
    X_fitur = df_scaled[kolom_fitur].values
    hasil_eval_km = evaluation.evaluasi_kmeans(X_fitur, label_cluster, inertia=inertia)
    evaluation.simpan_laporan_clustering(hasil_eval_km)

    # 4f. Visualize clustering results
    print("\n[SUB-STAGE 4F] Visualizing clustering results...")
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
# STAGE 5: Naive Bayes Classification (Distribution Areas)
# ============================================================
def tahap_naive_bayes(df_labeled, label_encoder_classes=None):
    """Classify distribution areas using Naive Bayes."""
    cetak_banner("STAGE 5: DISTRIBUTION AREA CLASSIFICATION (NAIVE BAYES)")

    # 5a. Prepare training data
    print("[SUB-STAGE 5A] Preparing training data...")
    X, y, nama_fitur, label_encoder = naive_bayes_model.siapkan_data_training(
        df_labeled, kolom_label='cluster_label'
    )
    if X is None:
        print("[ERROR] Failed to prepare training data for Naive Bayes!")
        sys.exit(1)

    # 5b. Train Naive Bayes model
    print("\n[SUB-STAGE 5B] Training Gaussian Naive Bayes...")
    hasil_training = naive_bayes_model.training_naive_bayes(X, y)
    model_nb = hasil_training['model']
    naive_bayes_model.simpan_model_nb(model_nb)

    # 5c. Evaluate model
    print("\n[SUB-STAGE 5C] Evaluating Naive Bayes Classification...")
    label_names = config.get_cluster_names_list(label_encoder.classes_)
    hasil_eval_nb = evaluation.evaluasi_naive_bayes(
        hasil_training['y_test'],
        hasil_training['y_pred'],
        label_names=label_names
    )
    evaluation.simpan_laporan_klasifikasi(hasil_eval_nb)

    # 5d. Visualize confusion matrix
    print("\n[SUB-STAGE 5D] Confusion Matrix Visualization...")
    visualization.plot_confusion_matrix(
        hasil_training['y_test'],
        hasil_training['y_pred'],
        label_names=label_names
    )

    # 5e. Visualize per-class metrics
    print("\n[SUB-STAGE 5E] Per-Class Performance Visualization...")
    visualization.plot_per_class_metrics(
        hasil_training['y_test'],
        hasil_training['y_pred'],
        label_names=label_names
    )

    return hasil_training, hasil_eval_nb, label_encoder


# ============================================================
# STAGE 5B: Random Forest Classification for Coffee Bean Variety
# ============================================================
def tahap_naive_bayes_variety(df_fitur):
    """Classify coffee bean varieties using Random Forest."""
    cetak_banner("STAGE 5B: COFFEE BEAN VARIETY CLASSIFICATION (RANDOM FOREST)")

    # 5b-a. Prepare training data
    print("[SUB-STAGE 5B-A] Preparing training data for coffee varieties...")
    X, y, nama_fitur, label_encoder = naive_bayes_model.siapkan_data_training(
        df_fitur, kolom_label='jenis_kopi'
    )
    if X is None:
        print("[ERROR] Failed to prepare training data for Random Forest!")
        sys.exit(1)

    import joblib
    try:
        joblib.dump(label_encoder, config.VARIETY_LABEL_ENCODER_PATH)
        print(f"[OK] Variety LabelEncoder saved to: {config.VARIETY_LABEL_ENCODER_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save variety LabelEncoder: {e}")

    # 5b-b. Train Random Forest model for coffee varieties
    print("\n[SUB-STAGE 5B-B] Training Random Forest for coffee varieties...")
    hasil_training = naive_bayes_model.training_random_forest(
        X,
        y,
        feature_names=nama_fitur,
        save_scaler_path=config.VARIETY_SCALER_PATH,
        n_estimators=200
    )
    model_nb = hasil_training['model']
    
    naive_bayes_model.simpan_model_nb(model_nb, path_output=config.VARIETY_MODEL_PATH)

    # 5b-c. Evaluate model
    print("\n[SUB-STAGE 5B-C] Evaluating Random Forest Classification for varieties...")
    # Format labels as English variety names (Arabica, Liberica, Robusta)
    variety_map = {'arabika': 'Arabica', 'liberika': 'Liberica', 'robusta': 'Robusta'}
    label_names = [variety_map.get(str(c).lower(), str(c).capitalize()) for c in label_encoder.classes_]
    
    hasil_eval_nb = evaluation.evaluasi_naive_bayes(
        hasil_training['y_test'],
        hasil_training['y_pred'],
        label_names=label_names
    )
    evaluation.simpan_laporan_klasifikasi(hasil_eval_nb, path_output=config.VARIETY_CLASSIFICATION_REPORT_PATH)

    # 5b-d. Confusion matrix visualization for coffee bean variety
    print("\n[SUB-STAGE 5B-D] Confusion Matrix Visualization - Coffee Bean Variety...")
    visualization.plot_confusion_matrix(
        hasil_training['y_test'],
        hasil_training['y_pred'],
        label_names=label_names,
        path_output=config.VARIETY_CONFUSION_MATRIX_PATH,
        title='Confusion Matrix - Coffee Bean Variety Classification\n(Random Forest)'
    )

    # 5b-e. Per-class performance metrics for coffee bean variety
    print("\n[SUB-STAGE 5B-E] Per-Class Performance Visualization - Coffee Bean Variety...")
    visualization.plot_per_class_metrics(
        hasil_training['y_test'],
        hasil_training['y_pred'],
        label_names=label_names,
        path_output=config.VARIETY_PER_CLASS_PATH,
        title_prefix='Coffee Bean Variety'
    )

    return hasil_training, hasil_eval_nb, label_encoder


# ============================================================
# STAGE 6: Summary & Output
# ============================================================
def tahap_ringkasan(hasil_eval_km, hasil_eval_nb, hasil_eval_variety=None):
    """Display final summary of the entire pipeline."""
    cetak_banner("STAGE 6: RESULTS SUMMARY")

    print("  >> PIPELINE RESULTS SUMMARY")
    print("  " + "-" * 50)

    # K-Means summary
    print("\n  [K-Means Clustering]")
    print(f"  Number of Distribution Areas: {hasil_eval_km['jumlah_cluster']}")
    print(f"  Silhouette Score            : {hasil_eval_km['silhouette_score']:.4f}")
    print(f"  Inertia                     : {hasil_eval_km['inertia']}")
    print("  Distribution per Area:")
    for cluster, jumlah in hasil_eval_km['distribusi'].items():
        nama = config.get_cluster_name(cluster)
        print(f"    {nama:<18} : {jumlah} samples")

    # Naive Bayes Area summary
    print("\n  [Naive Bayes Classification (Distribution Areas)]")
    print(f"  Accuracy            : {hasil_eval_nb['akurasi']:.4f} ({hasil_eval_nb['akurasi']*100:.2f}%)")
    print(f"  Precision           : {hasil_eval_nb['presisi']:.4f}")
    print(f"  Recall              : {hasil_eval_nb['recall']:.4f}")
    print(f"  F1-Score            : {hasil_eval_nb['f1_score']:.4f}")

    # Random Forest Variety summary
    if hasil_eval_variety is not None:
        print("\n  [Random Forest Classification (Coffee Bean Variety)]")
        print(f"  Accuracy            : {hasil_eval_variety['akurasi']:.4f} ({hasil_eval_variety['akurasi']*100:.2f}%)")
        print(f"  Precision           : {hasil_eval_variety['presisi']:.4f}")
        print(f"  Recall              : {hasil_eval_variety['recall']:.4f}")
        print(f"  F1-Score            : {hasil_eval_variety['f1_score']:.4f}")

    # Output file list
    print("\n  [Generated Output Files]")
    print("  " + "-" * 50)
    file_output = [
        ("K-Means Model", config.KMEANS_MODEL_PATH),
        ("Naive Bayes Model (Area)", config.NAIVE_BAYES_MODEL_PATH),
        ("Variety Model", config.VARIETY_MODEL_PATH),
        ("Scaler", config.SCALER_PATH),
        ("Variety Scaler", config.VARIETY_SCALER_PATH),
        ("Variety Label Encoder", config.VARIETY_LABEL_ENCODER_PATH),
        ("Extracted Features", config.FEATURES_EXTRACTED_CSV),
        ("Scaled Features", config.FEATURES_SCALED_CSV),
        ("Labeled Dataset", config.LABELED_DATASET_CSV),
        ("Elbow Curve", config.ELBOW_CURVE_PATH),
        ("Silhouette per K", config.SILHOUETTE_PER_K_PATH),
        ("Feature Distribution", config.FEATURE_DISTRIBUTION_PATH),
        ("Confusion Matrix (Area)", config.CONFUSION_MATRIX_PATH),
        ("Confusion Matrix (Variety)", config.VARIETY_CONFUSION_MATRIX_PATH),
        ("PCA Scatter 2D", config.PCA_SCATTER_PATH),
        ("Correlation Heatmap", config.CORRELATION_HEATMAP_PATH),
        ("Silhouette Analysis", config.SILHOUETTE_ANALYSIS_PATH),
        ("RGB Distribution", config.RGB_DISTRIBUTION_PATH),
        ("Area Size Chart", config.CLUSTER_SIZE_PATH),
        ("Radar Chart (Area)", config.RADAR_CHART_PATH),
        ("Pair Plot", config.PAIR_PLOT_PATH),
        ("Metric Comparison", config.METRIC_COMPARISON_PATH),
        ("Per-Area Metrics", config.CLASSIFICATION_PER_CLASS_PATH),
        ("Per-Variety Metrics", config.VARIETY_PER_CLASS_PATH),
        ("Clustering Report", config.CLUSTERING_REPORT_PATH),
        ("Classification Report (Area)", config.CLASSIFICATION_REPORT_PATH),
        ("Classification Report (Variety)", config.VARIETY_CLASSIFICATION_REPORT_PATH),
    ]
    for nama, path in file_output:
        status = "[OK]" if os.path.exists(path) else "[--]"
        print(f"    {status} {nama}: {os.path.basename(path)}")

    print("\n" + "=" * 60)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# ============================================================
# MAIN FUNCTION: Run Full Pipeline
# ============================================================
def main():
    """Main function executing the pipeline sequentially."""

    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 58 + "|")
    print("|" + "DISTRIBUTION AREA MAPPING & COFFEE BEAN QUALITY".center(58) + "|")
    print("|" + "CLASSIFICATION USING K-MEANS & NAIVE BAYES".center(58) + "|")
    print("|" + " " * 58 + "|")
    print("+" + "=" * 58 + "+")

    # STAGE 0: Initialization
    tahap_inisialisasi()

    # STAGE 1: Data Loading
    list_gambar, list_label, list_nama, df_kualitas, df_wilayah = tahap_muat_data()

    # STAGE 2: Image Preprocessing
    list_hasil_prep = tahap_preprocessing(list_gambar)

    # STAGE 3: Feature Extraction
    df_fitur = tahap_ekstraksi_fitur(list_hasil_prep, list_nama, list_label=list_label)

    # STAGE 4: K-Means Clustering
    df_scaled, df_labeled, scaler, label_cluster, hasil_eval_km, hasil_elbow = tahap_kmeans_clustering(df_fitur)

    # STAGE 5: Naive Bayes Classification (Distribution Areas)
    hasil_training_nb, hasil_eval_nb, label_encoder = tahap_naive_bayes(df_labeled)

    # STAGE 5B: Random Forest Classification (Coffee Variety)
    hasil_training_variety, hasil_eval_variety, label_encoder_variety = tahap_naive_bayes_variety(df_fitur)

    # STAGE 5E: Full Statistical Visualizations
    cetak_banner("STAGE 5E: STATISTICAL VISUALIZATION FOR DISTRIBUTION AREAS")
    kolom_fitur_nb = [c for c in df_scaled.columns if c not in ['nama_file', 'jenis_kopi']]
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

    # STAGE 6: Summary
    tahap_ringkasan(hasil_eval_km, hasil_eval_nb, hasil_eval_variety=hasil_eval_variety)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()

