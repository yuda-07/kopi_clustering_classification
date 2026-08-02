"""
evaluation.py - Modul Evaluasi Model
======================================
Menyediakan fungsi evaluasi untuk kedua model:
- K-Means: Silhouette Score, distribusi cluster, inertia
- Naive Bayes: Akurasi, Precision, Recall, F1, Confusion Matrix
- Mencetak laporan ke console dan menyimpan ke file .txt
"""

import os
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================
# EVALUASI K-MEANS CLUSTERING
# ============================================================

def evaluasi_kmeans(X_fitur, label_cluster, inertia=None):
    """
    Mengevaluasi hasil clustering K-Means.

    Parameter:
        X_fitur (np.ndarray): Data fitur yang digunakan untuk clustering.
        label_cluster (np.ndarray): Label cluster hasil K-Means.
        inertia (float): Nilai inertia dari model K-Means (opsional).

    Return:
        dict: Hasil evaluasi clustering.
    """
    print("=" * 50)
    print("  K-MEANS CLUSTERING EVALUATION")
    print("=" * 50)

    # Calculate Silhouette Score
    sil_score = silhouette_score(X_fitur, label_cluster)
    print(f"  Silhouette Score : {sil_score:.4f}")

    # Data distribution per area
    unique_labels, counts = np.unique(label_cluster, return_counts=True)
    print(f"  Number of distribution areas : {len(unique_labels)}")
    print(f"  Data distribution            :")
    for label, count in zip(unique_labels, counts):
        persentase = (count / len(label_cluster)) * 100
        nama = config.get_cluster_name(label)
        print(f"    {nama:<18}: {count} samples ({persentase:.1f}%)")

    # Inertia (if available)
    if inertia is not None:
        print(f"  Inertia                      : {inertia:.2f}")

    hasil = {
        'silhouette_score': sil_score,
        'jumlah_cluster': len(unique_labels),
        'distribusi': dict(zip(unique_labels.tolist(), counts.tolist())),
        'inertia': inertia,
        'total_data': len(label_cluster),
    }

    print("=" * 50)
    return hasil


def simpan_laporan_clustering(hasil_evaluasi, path_output=None):
    """
    Saves clustering evaluation report to a .txt file.

    Parameters:
        hasil_evaluasi (dict): Result from evaluasi_kmeans().
        path_output (str): Output file path (default from config).
    """
    if path_output is None:
        path_output = config.CLUSTERING_REPORT_PATH

    try:
        with open(path_output, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  K-MEANS CLUSTERING EVALUATION REPORT\n")
            f.write("  Coffee Bean Distribution Area Grouping\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total Samples               : {hasil_evaluasi['total_data']}\n")
            f.write(f"Number of Distribution Areas: {hasil_evaluasi['jumlah_cluster']}\n")
            f.write(f"Silhouette Score            : {hasil_evaluasi['silhouette_score']:.4f}\n")

            if hasil_evaluasi['inertia'] is not None:
                f.write(f"Inertia                     : {hasil_evaluasi['inertia']:.2f}\n")

            f.write(f"\nData Distribution per Area:\n")
            f.write("-" * 40 + "\n")
            for cluster, jumlah in hasil_evaluasi['distribusi'].items():
                pct = (jumlah / hasil_evaluasi['total_data']) * 100
                nama = config.get_cluster_name(cluster)
                f.write(f"  {nama:<18} : {jumlah} samples ({pct:.1f}%)\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("  Metric Notes:\n")
            f.write("  - Silhouette Score close to 1 = excellent clustering separation\n")
            f.write("  - Silhouette Score close to 0 = overlapping clusters\n")
            f.write("  - Negative Silhouette Score   = potential incorrect cluster assignment\n")
            f.write("=" * 60 + "\n")

        print(f"[OK] Clustering report saved to: {path_output}")

    except Exception as e:
        print(f"[ERROR] Failed to save clustering report: {e}")


# ============================================================
# NAIVE BAYES CLASSIFICATION EVALUATION
# ============================================================

def evaluasi_naive_bayes(y_test, y_pred, label_names=None):
    """
    Evaluates classification model results in detail.
    """
    print("=" * 50)
    print("  CLASSIFICATION MODEL EVALUATION")
    print("=" * 50)

    # Calculate all metrics
    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_val = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_val = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy  : {akurasi:.4f} ({akurasi*100:.2f}%)")
    print(f"  Precision : {presisi:.4f}")
    print(f"  Recall    : {recall_val:.4f}")
    print(f"  F1-Score  : {f1_val:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

    # Display classification report
    print(f"\n  Classification Report:")
    labels_present = sorted(set(y_test.tolist() + y_pred.tolist()))
    if label_names and len(label_names) == len(labels_present):
        report_str = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
    else:
        report_str = classification_report(y_test, y_pred, labels=labels_present, zero_division=0)
    print(report_str)

    # Per-class metrics
    p, r, f1, sup = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    per_class = {}
    for i, lbl in enumerate(labels_present):
        key = label_names[i] if (label_names and len(label_names) == len(labels_present)) else str(lbl)
        per_class[key] = {
            'precision': float(p[i]),
            'recall': float(r[i]),
            'f1': float(f1[i]),
            'support': int(sup[i])
        }

    hasil = {
        'akurasi': akurasi,
        'presisi': presisi,
        'recall': recall_val,
        'f1_score': f1_val,
        'confusion_matrix': cm,
        'classification_report': report_str,
        'per_class': per_class,
        'label_names': label_names if label_names else [str(l) for l in labels_present],
    }

    print("=" * 50)
    return hasil


def simpan_laporan_klasifikasi(hasil_evaluasi, path_output=None):
    """
    Saves classification evaluation report to a .txt file.
    """
    if path_output is None:
        path_output = config.CLASSIFICATION_REPORT_PATH

    try:
        with open(path_output, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("  CLASSIFICATION MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Accuracy    : {hasil_evaluasi['akurasi']:.4f} ({hasil_evaluasi['akurasi']*100:.2f}%)\n")
            f.write(f"Precision   : {hasil_evaluasi['presisi']:.4f}\n")
            f.write(f"Recall      : {hasil_evaluasi['recall']:.4f}\n")
            f.write(f"F1-Score    : {hasil_evaluasi['f1_score']:.4f}\n")

            f.write(f"\nConfusion Matrix:\n")
            f.write(str(hasil_evaluasi['confusion_matrix']) + "\n")

            f.write(f"\nClassification Report:\n")
            f.write(hasil_evaluasi['classification_report'] + "\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("  Metric Interpretation:\n")
            f.write("  - Accuracy  : Overall proportion of correct predictions\n")
            f.write("  - Precision : Proportion of correct positive predictions\n")
            f.write("  - Recall    : Proportion of actual positive instances correctly identified\n")
            f.write("  - F1-Score  : Harmonic mean of Precision and Recall\n")
            f.write("=" * 60 + "\n")

        print(f"[OK] Classification report saved to: {path_output}")

    except Exception as e:
        print(f"[ERROR] Failed to save classification report: {e}")


# ============================================================
# TEST MODUL (jalankan langsung)
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: evaluation.py")
    print("=" * 50)

    # Data dummy untuk K-Means
    np.random.seed(42)
    X_dummy_km = np.random.randn(100, 5)
    labels_dummy = np.random.choice([0, 1, 2], size=100)
    hasil_km = evaluasi_kmeans(X_dummy_km, labels_dummy, inertia=500.0)
    simpan_laporan_clustering(hasil_km)

    # Data dummy untuk Naive Bayes
    y_test_dummy = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    y_pred_dummy = np.array([0, 0, 1, 1, 2, 1, 0, 2, 2, 0])
    hasil_nb = evaluasi_naive_bayes(y_test_dummy, y_pred_dummy)
    simpan_laporan_klasifikasi(hasil_nb)

    print("=" * 50)
    print("Test evaluation selesai.")
