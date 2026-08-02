"""
kmeans_model.py - Modul K-Means Clustering
============================================
Melakukan clustering pada fitur yang sudah diekstraksi menggunakan algoritma K-Means:
- Normalisasi fitur dengan StandardScaler
- Menentukan jumlah cluster optimal K menggunakan Elbow Method
- Training model KMeans dengan K optimal
- Evaluasi menggunakan Silhouette Score
- Simpan model ke file .pkl
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def normalisasi_fitur(df_fitur):
    """
    Melakukan standardisasi fitur numerik menggunakan StandardScaler
    (mean=0, std=1 per kolom fitur).

    Parameter:
        df_fitur (pd.DataFrame): DataFrame fitur mentah.

    Return:
        tuple: (df_scaled: pd.DataFrame, scaler: StandardScaler)
    """
    try:
        # Pisahkan kolom non-numerik (nama_file & jenis_kopi) dari kolom fitur
        kolom_non_fitur = [col for col in ['nama_file', 'jenis_kopi'] if col in df_fitur.columns]
        kolom_fitur = [c for c in df_fitur.columns if c not in kolom_non_fitur]

        # Inisialisasi scaler
        scaler = StandardScaler()

        # Fit dan transformasi data fitur
        data_scaled = scaler.fit_transform(df_fitur[kolom_fitur])

        # Buat DataFrame baru dengan nama kolom yang sama
        df_scaled = pd.DataFrame(data_scaled, columns=kolom_fitur)

        # Tambahkan kembali kolom non-fitur secara berurutan
        for idx, col in enumerate(kolom_non_fitur):
            if col in df_fitur.columns:
                df_scaled.insert(idx, col, df_fitur[col].values)

        n_fitur_numerik = df_scaled.shape[1] - len(kolom_non_fitur)
        print(f"[OK] Feature normalization successful: {n_fitur_numerik} numeric features scaled.")
        return df_scaled, scaler

    except Exception as e:
        print(f"[ERROR] Failed to normalize features: {e}")
        return None, None


def hapus_outlier(df_scaled, threshold_z=3.5):
    """
    Removes outlier data based on Z-score.
    """
    kolom_non_fitur = ['nama_file', 'jenis_kopi']
    kolom_fitur = [c for c in df_scaled.columns if c not in kolom_non_fitur]
    data_num = df_scaled[kolom_fitur].values

    z_scores = np.abs(data_num)

    mask = (z_scores <= threshold_z).all(axis=1)
    n_outlier = (~mask).sum()
    df_bersih = df_scaled[mask].reset_index(drop=True)

    print(f"[OK] Outlier detection: {n_outlier} outlier samples removed (threshold z={threshold_z}).")
    print(f"     Remaining data: {len(df_bersih)} out of {len(df_scaled)}.")

    return df_bersih


def simpan_scaler(scaler, path_output=None):
    """
    Saves StandardScaler object to a file.
    """
    if path_output is None:
        path_output = config.SCALER_PATH
    try:
        joblib.dump(scaler, path_output)
        print(f"[OK] Scaler saved successfully to: {path_output}")
    except Exception as e:
        print(f"[ERROR] Failed to save scaler: {e}")


def elbow_method(df_scaled, k_min=None, k_max=None):
    """
    Executes K-Means for various K values and calculates inertia to determine optimal K.
    """
    if k_min is None:
        k_min = config.K_MIN
    if k_max is None:
        k_max = config.K_MAX

    kolom_non_fitur = ['nama_file', 'jenis_kopi']
    kolom_fitur = [c for c in df_scaled.columns if c not in kolom_non_fitur]
    X = df_scaled[kolom_fitur].values

    nilai_k = list(range(k_min, k_max + 1))
    inertia_list = []
    silhouette_scores = []

    if config.OPTIMAL_K is not None:
        k_optimal = config.OPTIMAL_K
        print(f"[INFO] OPTIMAL_K is set in config: K = {k_optimal}")
        print(f"[INFO] Running Elbow Method for K = {k_min} to {k_max} (for visualization)...")
    else:
        k_optimal = None
        print(f"[INFO] Running Elbow Method for K = {k_min} to {k_max}...")

    for k in nilai_k:
        model_km = KMeans(
            n_clusters=k,
            max_iter=config.KMEANS_MAX_ITER,
            n_init=config.KMEANS_N_INIT,
            random_state=config.RANDOM_STATE
        )
        model_km.fit(X)

        inertia_list.append(model_km.inertia_)
        sil_score = silhouette_score(X, model_km.labels_)
        silhouette_scores.append(sil_score)
        print(f"     K={k:2d} | Inertia={model_km.inertia_:.2f} | Silhouette={sil_score:.4f}")

    if k_optimal is not None:
        print(f"[OK] Using fixed K = {k_optimal} (from config)")
        return {
            'nilai_k': nilai_k,
            'inertia': inertia_list,
            'silhouette_scores': silhouette_scores,
            'k_optimal': k_optimal,
        }

    n_total = len(X)
    best_score = -1

    for idx, k in enumerate(nilai_k):
        km_check = KMeans(
            n_clusters=k, max_iter=config.KMEANS_MAX_ITER,
            n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_STATE
        )
        labels_check = km_check.fit_predict(X)
        _, counts = np.unique(labels_check, return_counts=True)
        min_cluster_pct = (counts.min() / n_total) * 100

        if min_cluster_pct < 5.0:
            print(f"     -> K={k} rejected: smallest cluster is only {min_cluster_pct:.1f}%")
            continue

        if silhouette_scores[idx] > best_score:
            best_score = silhouette_scores[idx]
            k_optimal = k

    if k_optimal is None:
        k_optimal = 3
        print(f"[INFO] All K produced imbalanced clusters, fallback to K={k_optimal}")
    else:
        print(f"[OK] Optimal K based on Silhouette Score: K = {k_optimal}")

    return {
        'nilai_k': nilai_k,
        'inertia': inertia_list,
        'silhouette_scores': silhouette_scores,
        'k_optimal': k_optimal,
    }


def training_kmeans(df_scaled, n_clusters=None):
    """
    Trains K-Means model with specified number of clusters.
    """
    if n_clusters is None:
        n_clusters = config.OPTIMAL_K if config.OPTIMAL_K else 3

    kolom_non_fitur = ['nama_file', 'jenis_kopi']
    kolom_fitur = [c for c in df_scaled.columns if c not in kolom_non_fitur]
    X = df_scaled[kolom_fitur].values

    print(f"[INFO] Training K-Means with K={n_clusters} clusters...")

    model = KMeans(
        n_clusters=n_clusters,
        max_iter=config.KMEANS_MAX_ITER,
        n_init=config.KMEANS_N_INIT,
        random_state=config.RANDOM_STATE
    )
    model.fit(X)

    label_cluster = model.labels_
    inertia = model.inertia_

    sil_score = silhouette_score(X, label_cluster)

    unique, counts = np.unique(label_cluster, return_counts=True)
    print(f"[OK] K-Means training complete.")
    print(f"     Inertia           : {inertia:.2f}")
    print(f"     Silhouette Score  : {sil_score:.4f}")
    print(f"     Area Distribution :")
    for u, c in zip(unique, counts):
        nama = config.get_cluster_name(u)
        print(f"       {nama:<16}: {c} samples")

    return model, label_cluster, inertia, sil_score


def simpan_model_kmeans(model, path_output=None):
    """
    Saves K-Means model to a .pkl file.
    """
    if path_output is None:
        path_output = config.KMEANS_MODEL_PATH
    try:
        joblib.dump(model, path_output)
        print(f"[OK] K-Means model saved to: {path_output}")
    except Exception as e:
        print(f"[ERROR] Failed to save K-Means model: {e}")


def buat_dataset_berlabel(df_scaled, label_cluster):
    """
    Appends cluster labels to features DataFrame and saves it.
    """
    df_labeled = df_scaled.copy()
    df_labeled['cluster_label'] = label_cluster

    try:
        df_labeled.to_csv(config.LABELED_DATASET_CSV, index=False)
        print(f"[OK] Labeled dataset saved to: {config.LABELED_DATASET_CSV}")
    except Exception as e:
        print(f"[ERROR] Failed to save labeled dataset: {e}")

    return df_labeled


def muat_model_kmeans(path_model=None):
    """
    Loads K-Means model from a .pkl file.
    """
    if path_model is None:
        path_model = config.KMEANS_MODEL_PATH
    try:
        model = joblib.load(path_model)
        print(f"[OK] K-Means model loaded from: {path_model}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load K-Means model: {e}")
        return None


# ============================================================
# TEST MODUL (jalankan langsung)
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: kmeans_model.py")
    print("=" * 50)

    # Buat data dummy
    np.random.seed(config.RANDOM_STATE)
    data_dummy = np.random.randn(100, 10)
    df_dummy = pd.DataFrame(data_dummy, columns=[f'fitur_{i}' for i in range(10)])
    df_dummy.insert(0, 'nama_file', [f'img_{i:03d}.jpg' for i in range(100)])

    # Normalisasi
    df_scaled, scaler = normalisasi_fitur(df_dummy)

    if df_scaled is not None:
        # Elbow Method
        hasil_elbow = elbow_method(df_scaled, k_min=2, k_max=5)
        k_opt = hasil_elbow['k_optimal']

        # Training dengan K optimal
        model, labels, inertia, sil = training_kmeans(df_scaled, n_clusters=k_opt)
        print(f"  Label unik: {np.unique(labels)}")

    print("=" * 50)
    print("Test kmeans_model selesai.")
