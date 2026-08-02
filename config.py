"""
config.py - Global Project Configuration
==========================================
Stores all parameters, file paths, and constants used
throughout the entire project pipeline.
"""

import os

# ============================================================
# BASE PROJECT PATH
# ============================================================
# Automatically get the project root path based on config.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# DATASET PATHS
# ============================================================
# Raw dataset folder (coffee bean images + CSV metadata)
RAW_DATA_DIR = os.path.join(BASE_DIR, "dataset", "raw")
COFFEE_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "coffee_images")
COFFEE_QUALITY_CSV = os.path.join(RAW_DATA_DIR, "coffee_quality.csv")
DISTRIBUTION_REGION_CSV = os.path.join(RAW_DATA_DIR, "distribution_region.csv")

# Processed dataset folder (features & labels)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed")
FEATURES_EXTRACTED_CSV = os.path.join(PROCESSED_DATA_DIR, "features_extracted.csv")
FEATURES_SCALED_CSV = os.path.join(PROCESSED_DATA_DIR, "features_scaled.csv")
LABELED_DATASET_CSV = os.path.join(PROCESSED_DATA_DIR, "labeled_dataset.csv")

# ============================================================
# MODEL PATHS
# ============================================================
# Folder for storing trained models
MODELS_DIR = os.path.join(BASE_DIR, "models")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
NAIVE_BAYES_MODEL_PATH = os.path.join(MODELS_DIR, "naive_bayes_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Model paths for coffee bean variety classification (Arabica, Liberica, Robusta)
VARIETY_MODEL_PATH = os.path.join(MODELS_DIR, "variety_classifier_model.pkl")
VARIETY_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "variety_label_encoder.pkl")
VARIETY_SCALER_PATH = os.path.join(MODELS_DIR, "variety_scaler.pkl")

# ============================================================
# OUTPUT & VISUALIZATION PATHS
# ============================================================
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

# Main visualization output files
ELBOW_CURVE_PATH = os.path.join(PLOTS_DIR, "elbow_curve.png")
CLUSTER_MAP_PATH = os.path.join(PLOTS_DIR, "cluster_map.png")
CONFUSION_MATRIX_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
FEATURE_DISTRIBUTION_PATH = os.path.join(PLOTS_DIR, "feature_distribution.png")
SAMPLE_IMAGES_PATH = os.path.join(PLOTS_DIR, "sample_images.png")
SILHOUETTE_PER_K_PATH = os.path.join(PLOTS_DIR, "silhouette_scores.png")

# Variety classification visualization output files
VARIETY_CONFUSION_MATRIX_PATH = os.path.join(PLOTS_DIR, "variety_confusion_matrix.png")
VARIETY_PER_CLASS_PATH = os.path.join(PLOTS_DIR, "variety_per_class_metrics.png")

# Additional statistical plot output files
PCA_SCATTER_PATH = os.path.join(PLOTS_DIR, "pca_scatter_2d.png")
CORRELATION_HEATMAP_PATH = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
SILHOUETTE_ANALYSIS_PATH = os.path.join(PLOTS_DIR, "silhouette_analysis.png")
RGB_DISTRIBUTION_PATH = os.path.join(PLOTS_DIR, "rgb_distribution_per_type.png")
CLUSTER_SIZE_PATH = os.path.join(PLOTS_DIR, "cluster_size_bar.png")
RADAR_CHART_PATH = os.path.join(PLOTS_DIR, "radar_chart_cluster.png")
PAIR_PLOT_PATH = os.path.join(PLOTS_DIR, "pair_plot_top_features.png")
METRIC_COMPARISON_PATH = os.path.join(PLOTS_DIR, "metric_comparison.png")
CLASSIFICATION_PER_CLASS_PATH = os.path.join(PLOTS_DIR, "classification_per_class_metrics.png")

# Visualization display configuration
SHOW_PLOTS = True  # True = show popup window, False = save PNG only

# Report output files
CLUSTERING_REPORT_PATH = os.path.join(OUTPUTS_DIR, "clustering_report.txt")
CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "classification_report.txt")
VARIETY_CLASSIFICATION_REPORT_PATH = os.path.join(OUTPUTS_DIR, "variety_classification_report.txt")
PREDICTIONS_EXCEL_PATH = os.path.join(OUTPUTS_DIR, "predictions.xlsx")

# ============================================================
# IMAGE PREPROCESSING PARAMETERS
# ============================================================
# Image resize target (pixels)
IMG_SIZE = 128

# Gaussian Blur kernel size for noise removal
BLUR_KERNEL_SIZE = (5, 5)

# ============================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================
# Number of histogram bins per color channel
HISTOGRAM_BINS = 32

# ============================================================
# K-MEANS CLUSTERING PARAMETERS
# ============================================================
# K value range for Elbow Method (K=2 to K_MAX)
K_MIN = 2
K_MAX = 10

# Maximum iterations for K-Means
KMEANS_MAX_ITER = 300

# Number of K-Means initializations (n_init)
KMEANS_N_INIT = 10

# Optimal number of clusters (8 clusters as per research abstract)
OPTIMAL_K = 8

# ============================================================
# NAIVE BAYES CLASSIFICATION PARAMETERS
# ============================================================
# Train/test split ratio
TEST_SIZE = 0.2

# ============================================================
# GENERAL PARAMETERS
# ============================================================
# Random state for all random operations (reproducibility)
RANDOM_STATE = 42

# List of coffee bean varieties (class labels based on dataset folders)
GRADE_LABELS = ["arabika", "liberika", "robusta"]

# ============================================================
# CLUSTER NAME MAPPING (replaces "Cluster 0", "Cluster 1", ...)
# 8 distribution area clusters based on visual characteristics (per research abstract)
# Names reflect the visual/textural properties of each group
# ============================================================
CLUSTER_NAME_MAP = {
    0: "Dark Smooth",
    1: "Light Pale",
    2: "Dark Coarse",
    3: "Medium Coarse",
    4: "Light Smooth",
    5: "Standard",
    6: "Very Dark",
    7: "Light Uniform"
}


def get_cluster_name(cluster_id):
    """Returns a descriptive name for the given cluster ID."""
    return CLUSTER_NAME_MAP.get(cluster_id, f"Cluster {cluster_id}")


def get_cluster_names_list(cluster_ids):
    """Returns a list of descriptive names in the order of cluster_ids."""
    return [get_cluster_name(cid) for cid in cluster_ids]


# Supported image file extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# ============================================================
# UTILITY FUNCTION: Ensure all required directories exist
# ============================================================
def ensure_directories():
    """
    Creates all required project directories if they do not exist.
    Called at the start of the pipeline to prevent path-not-found errors.
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
    print("[OK] All project directories are ready.")


# Run automatically when config is imported as main
if __name__ == "__main__":
    ensure_directories()
    print("[INFO] Project configuration loaded successfully.")
    print(f"[INFO] Base Dir : {BASE_DIR}")
    print(f"[INFO] Images  : {COFFEE_IMAGES_DIR}")
    print(f"[INFO] Models  : {MODELS_DIR}")
    print(f"[INFO] Outputs : {OUTPUTS_DIR}")
