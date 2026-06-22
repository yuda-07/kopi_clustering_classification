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
        # Pisahkan kolom non-numerik (nama_file) dari kolom fitur
        kolom_non_fitur = ['nama_file']
        kolom_fitur = [c for c in df_fitur.columns if c not in kolom_non_fitur]

        # Inisialisasi scaler
        scaler = StandardScaler()

        # Fit dan transformasi data fitur
        data_scaled = scaler.fit_transform(df_fitur[kolom_fitur])

        # Buat DataFrame baru dengan nama kolom yang sama
        df_scaled = pd.DataFrame(data_scaled, columns=kolom_fitur)

        # Tambahkan kembali kolom non-fitur
        for col in kolom_non_fitur:
            if col in df_fitur.columns:
                df_scaled.insert(0, col, df_fitur[col].values)

        print(f"[OK] Normalisasi fitur berhasil: {df_scaled.shape[1]-1} fitur numerik di-scale.")
        return df_scaled, scaler

    except Exception as e:
        print(f"[ERROR] Gagal normalisasi fitur: {e}")
        return None, None


def hapus_outlier(df_scaled, threshold_z=3.5):
    """
    Menghapus data outlier berdasarkan Z-score.
    Data dengan |z-score| > threshold di kolom fitur manapun akan dibuang.

    Parameter:
        df_scaled (pd.DataFrame): DataFrame fitur yang sudah di-scale.
        threshold_z (float): Batas Z-score untuk deteksi outlier.

    Return:
        pd.DataFrame: DataFrame tanpa outlier.
    """
    kolom_fitur = [c for c in df_scaled.columns if c != 'nama_file']
    data_num = df_scaled[kolom_fitur].values

    # Hitung Z-score absolut untuk setiap fitur
    z_scores = np.abs(data_num)  # Data sudah di-scale (mean=0, std=1)

    # Mask: True jika data BUKAN outlier
    mask = (z_scores <= threshold_z).all(axis=1)
    n_outlier = (~mask).sum()
    df_bersih = df_scaled[mask].reset_index(drop=True)

    print(f"[OK] Deteksi outlier: {n_outlier} data outlier dibuang (threshold z={threshold_z}).")
    print(f"     Data tersisa: {len(df_bersih)} dari {len(df_scaled)}.")

    return df_bersih


def simpan_scaler(scaler, path_output=None):
    """
    Menyimpan objek StandardScaler ke file.

    Parameter:
        scaler (StandardScaler): Objek scaler yang sudah fit.
        path_output (str): Path file output (default dari config).
    """
    if path_output is None:
        path_output = config.SCALER_PATH
    try:
        joblib.dump(scaler, path_output)
        print(f"[OK] Scaler berhasil disimpan ke: {path_output}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan scaler: {e}")


def elbow_method(df_scaled, k_min=None, k_max=None):
    """
    Menjalankan K-Means untuk berbagai nilai K dan menghitung inertia
    untuk menentukan jumlah cluster optimal dengan Elbow Method.

    Parameter:
        df_scaled (pd.DataFrame): DataFrame fitur yang sudah di-scale.
        k_min (int): Nilai K minimum (default dari config).
        k_max (int): Nilai K maksimum (default dari config).

    Return:
        dict: {
            'nilai_k': list[int],
            'inertia': list[float],
            'silhouette_scores': list[float],
            'k_optimal': int
        }
    """
    if k_min is None:
        k_min = config.K_MIN
    if k_max is None:
        k_max = config.K_MAX

    # Ambil hanya kolom numerik (buang nama_file)
    kolom_fitur = [c for c in df_scaled.columns if c != 'nama_file']
    X = df_scaled[kolom_fitur].values

    nilai_k = list(range(k_min, k_max + 1))
    inertia_list = []
    silhouette_scores = []

    print(f"[INFO] Menjalankan Elbow Method untuk K = {k_min} sampai {k_max}...")

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

    # Tentukan K optimal berdasarkan Silhouette Score tertinggi,
    # TAPI tolak K yang menghasilkan cluster terlalu kecil (< 5% dari total data)
    X = df_scaled[[c for c in df_scaled.columns if c != 'nama_file']].values
    n_total = len(X)
    k_optimal = None
    best_score = -1

    for idx, k in enumerate(nilai_k):
        # Quick check: jalankan KMeans dan cek distribusi cluster
        km_check = KMeans(
            n_clusters=k, max_iter=config.KMEANS_MAX_ITER,
            n_init=config.KMEANS_N_INIT, random_state=config.RANDOM_STATE
        )
        labels_check = km_check.fit_predict(X)
        _, counts = np.unique(labels_check, return_counts=True)
        min_cluster_pct = (counts.min() / n_total) * 100

        # Tolak jika cluster terkecil < 5% dari total data
        if min_cluster_pct < 5.0:
            print(f"     -> K={k} ditolak: cluster terkecil hanya {min_cluster_pct:.1f}%")
            continue

        if silhouette_scores[idx] > best_score:
            best_score = silhouette_scores[idx]
            k_optimal = k

    # Fallback: jika semua K ditolak, gunakan K=3 (default jumlah kelas alami)
    if k_optimal is None:
        k_optimal = min(3, len(nilai_k))
        print(f"[INFO] Semua K menghasilkan cluster tidak seimbang, fallback ke K={k_optimal}")
    else:
        print(f"[OK] K optimal berdasarkan Silhouette Score (cluster seimbang): K = {k_optimal}")

    return {
        'nilai_k': nilai_k,
        'inertia': inertia_list,
        'silhouette_scores': silhouette_scores,
        'k_optimal': k_optimal,
    }


def training_kmeans(df_scaled, n_clusters=None):
    """
    Melatih model K-Means dengan jumlah cluster yang ditentukan.

    Parameter:
        df_scaled (pd.DataFrame): DataFrame fitur yang sudah di-scale.
        n_clusters (int): Jumlah cluster K (default: K optimal dari config atau 3).

    Return:
        tuple: (model: KMeans, label_cluster: np.ndarray, inertia: float, sil_score: float)
    """
    if n_clusters is None:
        n_clusters = config.OPTIMAL_K if config.OPTIMAL_K else 3

    # Ambil hanya kolom numerik
    kolom_fitur = [c for c in df_scaled.columns if c != 'nama_file']
    X = df_scaled[kolom_fitur].values

    print(f"[INFO] Melatih K-Means dengan K={n_clusters} cluster...")

    # Inisialisasi dan training model K-Means
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=config.KMEANS_MAX_ITER,
        n_init=config.KMEANS_N_INIT,
        random_state=config.RANDOM_STATE
    )
    model.fit(X)

    label_cluster = model.labels_
    inertia = model.inertia_

    # Hitung Silhouette Score
    sil_score = silhouette_score(X, label_cluster)

    # Tampilkan distribusi cluster
    unique, counts = np.unique(label_cluster, return_counts=True)
    print(f"[OK] Training K-Means selesai.")
    print(f"     Inertia        : {inertia:.2f}")
    print(f"     Silhouette Score: {sil_score:.4f}")
    print(f"     Distribusi cluster:")
    for u, c in zip(unique, counts):
        print(f"       Cluster {u}: {c} data")

    return model, label_cluster, inertia, sil_score


def simpan_model_kmeans(model, path_output=None):
    """
    Menyimpan model K-Means ke file .pkl.

    Parameter:
        model (KMeans): Model K-Means yang sudah ditraining.
        path_output (str): Path file output (default dari config).
    """
    if path_output is None:
        path_output = config.KMEANS_MODEL_PATH
    try:
        joblib.dump(model, path_output)
        print(f"[OK] Model K-Means berhasil disimpan ke: {path_output}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan model K-Means: {e}")


def buat_dataset_berlabel(df_scaled, label_cluster):
    """
    Menambahkan label cluster ke DataFrame fitur dan menyimpannya.

    Parameter:
        df_scaled (pd.DataFrame): DataFrame fitur yang sudah di-scale.
        label_cluster (np.ndarray): Array label cluster per data.

    Return:
        pd.DataFrame: DataFrame dengan kolom 'cluster_label' tambahan.
    """
    df_labeled = df_scaled.copy()
    df_labeled['cluster_label'] = label_cluster

    try:
        df_labeled.to_csv(config.LABELED_DATASET_CSV, index=False)
        print(f"[OK] Dataset berlabel berhasil disimpan ke: {config.LABELED_DATASET_CSV}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan dataset berlabel: {e}")

    return df_labeled


def muat_model_kmeans(path_model=None):
    """
    Memuat model K-Means dari file .pkl.

    Parameter:
        path_model (str): Path file model (default dari config).

    Return:
        KMeans atau None: Model K-Means jika berhasil dimuat.
    """
    if path_model is None:
        path_model = config.KMEANS_MODEL_PATH
    try:
        model = joblib.load(path_model)
        print(f"[OK] Model K-Means berhasil dimuat dari: {path_model}")
        return model
    except Exception as e:
        print(f"[ERROR] Gagal memuat model K-Means: {e}")
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
