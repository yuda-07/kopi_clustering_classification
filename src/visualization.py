"""
visualization.py - Modul Visualisasi Lengkap
==============================================
Menyediakan semua fungsi plotting untuk proyek:

[PLOT UTAMA]
- Elbow Curve (inertia vs K)
- Silhouette Score per K
- Confusion Matrix heatmap
- Distribusi fitur per cluster
- Sampel gambar per grade/cluster

[PLOT STATISTIK BARU]
- PCA Scatter Plot 2D (visualisasi cluster di ruang 2D)
- Correlation Heatmap (korelasi antar fitur)
- Silhouette Analysis Plot (analisis per-sampel)
- RGB Channel Distribution per Jenis Kopi
- Wilayah Size Bar Chart (ukuran wilayah distribusi)
- Radar Chart (profil fitur per cluster)
- Pair Plot (scatter matrix fitur teratas)
- Metric Comparison (perbandingan metrik evaluasi)
- Per-Class Metrics (performa Precision/Recall/F1 per kelas klasifikasi)

Semua plot: TAMPILKAN POPUP + SIMPAN PNG
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix, silhouette_samples
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Gunakan backend yang mendukung popup (TkAgg default)
matplotlib.use('TkAgg')


# ============================================================
# HELPER: Ambil Warna Cluster dari Gambar Asli
# ============================================================
def ambil_warna_cluster_dari_gambar(labeled_csv_path=None, images_dir=None, n_sampel=30):
    """
    Menghitung warna rata-rata (mean RGB) dari gambar asli biji kopi
    untuk setiap cluster, lalu mengembalikannya sebagai dict warna matplotlib.

    Cara kerja:
      1. Baca labeled_dataset.csv → tahu gambar mana masuk cluster berapa
      2. Per cluster, ambil n_sampel gambar acak
      3. Baca pixel gambar asli (belum di-scale), hitung mean RGB
      4. Kembalikan sebagai tuple (R/255, G/255, B/255) per cluster

    Parameters
    ----------
    labeled_csv_path : str, opsional
        Path ke labeled_dataset.csv. Default: config.LABELED_DATASET_CSV
    images_dir : str, opsional
        Path ke folder coffee_images. Default: config.COFFEE_IMAGES_DIR
    n_sampel : int
        Jumlah gambar yang diambil sampel per cluster (default 30).

    Returns
    -------
    dict
        {cluster_id (int): (R, G, B) ternormalisasi ke [0, 1]}
    """
    if labeled_csv_path is None:
        labeled_csv_path = config.LABELED_DATASET_CSV
    if images_dir is None:
        images_dir = config.COFFEE_IMAGES_DIR

    # Baca CSV
    df = pd.read_csv(labeled_csv_path)

    # Pastikan kolom yang dibutuhkan ada
    if 'nama_file' not in df.columns or 'cluster_label' not in df.columns:
        print("[WARN] Kolom 'nama_file' atau 'cluster_label' tidak ditemukan. "
              "Fallback ke colormap Set3.")
        return None

    # Tentukan subfolder berdasarkan prefix nama file
    def _cari_path_gambar(nama_file):
        """Cari path lengkap gambar berdasarkan prefix nama file."""
        for subfolder in ['arabika', 'robusta', 'liberika']:
            path = os.path.join(images_dir, subfolder, nama_file)
            if os.path.exists(path):
                return path
        return None

    warna_per_cluster = {}
    cluster_ids = sorted(df['cluster_label'].unique())

    for cluster_id in cluster_ids:
        # Ambil semua baris yang masuk cluster ini
        baris_cluster = df[df['cluster_label'] == cluster_id]

        # Sampel acak (agar tidak terlalu lama)
        if len(baris_cluster) > n_sampel:
            baris_cluster = baris_cluster.sample(n=n_sampel, random_state=42)

        semua_pixel_rgb = []  # list untuk mengumpulkan mean RGB per gambar

        for nama_file in baris_cluster['nama_file'].values:
            path_gambar = _cari_path_gambar(nama_file)
            if path_gambar is None:
                continue

            # Baca gambar (OpenCV: BGR) → konversi ke RGB
            img_bgr = cv2.imread(path_gambar)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # -------------------------------------------------------
            # Filter piksel latar belakang (putih/terang)
            # Hanya pertahankan piksel yang brightness-nya < threshold
            # sehingga yang dihitung hanya piksel biji kopi asli.
            # Brightness = rata-rata R+G+B per piksel
            # -------------------------------------------------------
            pixels = img_rgb.reshape(-1, 3).astype(np.float32)
            brightness = pixels.mean(axis=1)          # brightness per piksel
            mask_kopi = brightness < 200              # True = piksel gelap (biji kopi)

            if mask_kopi.sum() < 10:
                # Hampir semua piksel terang (gambar terlalu terang/blank)
                # Turunkan threshold supaya ada data
                mask_kopi = brightness < 230

            pixels_kopi = pixels[mask_kopi]

            if len(pixels_kopi) == 0:
                continue

            mean_rgb = pixels_kopi.mean(axis=0)      # shape: (3,)
            semua_pixel_rgb.append(mean_rgb)

        if len(semua_pixel_rgb) == 0:
            print(f"[WARN] Cluster {cluster_id}: tidak ada gambar ditemukan. "
                  "Warna akan di-fallback ke Set3.")
            warna_per_cluster[cluster_id] = None
        else:
            # Rata-rata dari semua gambar di cluster ini → normalisasi ke [0, 1]
            mean_cluster = np.array(semua_pixel_rgb).mean(axis=0)
            warna_per_cluster[cluster_id] = tuple(mean_cluster / 255.0)

        mean_display = np.array(semua_pixel_rgb).mean(axis=0).astype(int) \
            if semua_pixel_rgb else 'N/A'
        print(f"[OK] Cluster {cluster_id}: mean RGB (tanpa bg) = {mean_display}")

    return warna_per_cluster


# ============================================================
# HELPER: Simpan + Tampilkan
# ============================================================
def _simpan_dan_tampilkan(fig, path_output, nama_plot):
    """
    Helper: simpan plot ke PNG dan tampilkan popup jika SHOW_PLOTS=True.
    """
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    plt.tight_layout()
    fig.savefig(path_output, dpi=150, bbox_inches='tight')
    print(f"[OK] {nama_plot} disimpan ke: {path_output}")

    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 1. ELBOW CURVE
# ============================================================
def plot_elbow_curve(nilai_k, inertia, k_optimal=None, path_output=None):
    """Memplot Elbow Curve: hubungan jumlah cluster K vs inertia."""
    if path_output is None:
        path_output = config.ELBOW_CURVE_PATH

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nilai_k, inertia, marker='o', linewidth=2, color='steelblue', label='Inertia')

    if k_optimal is not None:
        idx_opt = nilai_k.index(k_optimal)
        ax.scatter(k_optimal, inertia[idx_opt], color='red', s=150, zorder=5,
                   label=f'K Optimal = {k_optimal}')
        ax.annotate(f'K={k_optimal}', xy=(k_optimal, inertia[idx_opt]),
                    xytext=(k_optimal + 0.5, inertia[idx_opt]),
                    fontsize=12, color='red', fontweight='bold')

    ax.set_title('Elbow Method - Penentuan Jumlah Cluster Optimal', fontsize=14, fontweight='bold')
    ax.set_xlabel('Jumlah Cluster (K)', fontsize=12)
    ax.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax.set_xticks(nilai_k)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _simpan_dan_tampilkan(fig, path_output, "Elbow Curve")


# ============================================================
# 2. SILHOUETTE SCORE PER K
# ============================================================
def plot_silhouette_per_k(nilai_k, silhouette_scores, path_output=None):
    """Memplot Silhouette Score untuk setiap nilai K."""
    if path_output is None:
        path_output = config.SILHOUETTE_PER_K_PATH

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nilai_k, silhouette_scores, marker='s', linewidth=2, color='darkorange', label='Silhouette Score')

    # Tandai skor tertinggi
    idx_best = np.argmax(silhouette_scores)
    k_best = nilai_k[idx_best]
    ax.scatter(k_best, silhouette_scores[idx_best], color='green', s=150, zorder=5,
               label=f'Terbaik K={k_best} ({silhouette_scores[idx_best]:.4f})')

    ax.set_title('Silhouette Score per Jumlah Cluster (K)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Jumlah Cluster (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_xticks(nilai_k)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _simpan_dan_tampilkan(fig, path_output, "Silhouette Score per K")


# ============================================================
# 3. CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(y_test, y_pred, label_names=None, path_output=None, title=None):
    """Memplot Confusion Matrix sebagai heatmap."""
    if path_output is None:
        path_output = config.CONFUSION_MATRIX_PATH

    if title is None:
        title = 'Confusion Matrix - Klasifikasi Wilayah Distribusi\n(Naive Bayes)'

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediksi Wilayah', fontsize=12)
    ax.set_ylabel('Wilayah Sebenarnya', fontsize=12)

    _simpan_dan_tampilkan(fig, path_output, "Confusion Matrix")


# ============================================================
# 4. DISTRIBUSI FITUR PER CLUSTER (BOXPLOT)
# ============================================================
def plot_distribusi_cluster(df_scaled, label_cluster, fitur_pilih=None, path_output=None):
    """Memplot distribusi fitur per cluster menggunakan boxplot."""
    if path_output is None:
        path_output = config.FEATURE_DISTRIBUTION_PATH

    kolom_non_fitur = ['nama_file', 'jenis_kopi', 'cluster_label']
    kolom_fitur = [c for c in df_scaled.columns if c not in kolom_non_fitur]
    if fitur_pilih is None:
        fitur_pilih = kolom_fitur[:6]

    df_plot = df_scaled[fitur_pilih].copy()
    df_plot['cluster'] = label_cluster

    n_fitur = len(fitur_pilih)
    n_cols = 3
    n_rows = (n_fitur + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for i, fitur in enumerate(fitur_pilih):
        ax = axes[i // n_cols][i % n_cols] if n_rows > 1 else axes[i]
        sns.boxplot(x='cluster', y=fitur, hue='cluster', data=df_plot, ax=ax,
                    palette='Set2', legend=False)
        ax.set_title(f'Distribusi: {fitur}', fontsize=11)
        ax.set_xlabel('Cluster')
        ax.set_ylabel(fitur)

    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols] if n_rows > 1 else axes[j])

    fig.suptitle('Distribusi Fitur per Wilayah Distribusi', fontsize=14, fontweight='bold', y=1.02)

    _simpan_dan_tampilkan(fig, path_output, "Distribusi Fitur per Cluster")


# ============================================================
# 5. SAMPEL GAMBAR
# ============================================================
def plot_sampel_gambar(list_gambar_rgb, list_label, grade_nama=None,
                       max_sampel=9, path_output=None):
    """Memplot sampel gambar biji kopi dalam grid."""
    if path_output is None:
        path_output = config.SAMPLE_IMAGES_PATH

    n = min(len(list_gambar_rgb), max_sampel)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i // n_cols][i % n_cols] if n_rows > 1 else axes[i]
        ax.imshow(list_gambar_rgb[i])
        ax.set_title(f'Label: {list_label[i]}', fontsize=10)
        ax.axis('off')

    for j in range(n, n_rows * n_cols):
        ax = axes[j // n_cols][j % n_cols] if n_rows > 1 else axes[j]
        ax.axis('off')

    judul = 'Sampel Gambar Biji Kopi'
    if grade_nama:
        judul += f' - {grade_nama}'
    fig.suptitle(judul, fontsize=14, fontweight='bold')

    _simpan_dan_tampilkan(fig, path_output, "Sampel Gambar")


# ============================================================
# 6. PCA SCATTER PLOT 2D [BARU]
# ============================================================
def plot_pca_scatter(X_fitur, label_cluster, path_output=None):
    """
    Memplot scatter 2D menggunakan PCA untuk mereduksi dimensi fitur.
    Setiap titik = 1 data biji kopi, warna = cluster.
    """
    if path_output is None:
        path_output = config.PCA_SCATTER_PATH

    # Reduksi dimensi ke 2 komponen utama
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_fitur)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = np.unique(label_cluster)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = label_cluster == label
        cluster_name = config.get_cluster_name(label)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=cluster_name,
                   alpha=0.6, s=30, edgecolors='w', linewidth=0.3)

    # Tandai centroid di ruang PCA
    for label, color in zip(unique_labels, colors):
        mask = label_cluster == label
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.scatter(cx, cy, c=[color], marker='X', s=200, edgecolors='black', linewidth=2)

    ax.set_title(f'PCA Scatter Plot 2D - Visualisasi 8 Wilayah Distribusi\n'
                 f'(PC1={var1:.1f}% variance, PC2={var2:.1f}% variance)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Principal Component 1 ({var1:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({var2:.1f}%)', fontsize=12)
    ax.legend(loc='best', fontsize=9, title="Wilayah")
    ax.grid(True, alpha=0.2)

    _simpan_dan_tampilkan(fig, path_output, "PCA Scatter 2D")


# ============================================================
# 7. CORRELATION HEATMAP [BARU]
# ============================================================
def plot_correlation_heatmap(df_scaled, max_fitur=20, path_output=None):
    """
    Memplot heatmap korelasi antar fitur.
    Menampilkan max_fitur fitur dengan varians tertinggi.
    """
    if path_output is None:
        path_output = config.CORRELATION_HEATMAP_PATH

    kolom_non_fitur = ['nama_file', 'jenis_kopi', 'cluster_label']
    kolom_fitur = [c for c in df_scaled.columns if c not in kolom_non_fitur]

    # Pilih fitur dengan varians tertinggi agar heatmap tidak terlalu padat
    if len(kolom_fitur) > max_fitur:
        varians = df_scaled[kolom_fitur].var().sort_values(ascending=False)
        kolom_fitur = varians.head(max_fitur).index.tolist()

    corr_matrix = df_scaled[kolom_fitur].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, fmt='.2f',
                square=True, linewidths=0.5, ax=ax)
    ax.set_title(f'Korelasi Antar Fitur (Top {len(kolom_fitur)} Fitur)', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    _simpan_dan_tampilkan(fig, path_output, "Correlation Heatmap")


# ============================================================
# 8. SILHOUETTE ANALYSIS PLOT [BARU]
# ============================================================
def plot_silhouette_analysis(X_fitur, label_cluster, path_output=None):
    """
    Memplot Silhouette Analysis (plot "pisau") per sampel.
    Menunjukkan seberapa baik setiap data berada di cluster-nya.
    """
    if path_output is None:
        path_output = config.SILHOUETTE_ANALYSIS_PATH

    n_clusters = len(np.unique(label_cluster))
    silhouette_avg = silhouette_samples(X_fitur, label_cluster).mean()
    sample_silhouette_values = silhouette_samples(X_fitur, label_cluster)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_lower = 10
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for idx_cluster in range(n_clusters):
        # Ambil silhouette values untuk cluster ini
        mask = label_cluster == idx_cluster
        cluster_silhouette = sample_silhouette_values[mask]
        cluster_silhouette.sort()

        y_upper = y_lower + len(cluster_silhouette)
        color = colors[idx_cluster]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette),
                str(idx_cluster), fontsize=11, fontweight='bold')
        y_lower = y_upper + 10

    # Garis rata-rata silhouette
    ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
               label=f'Rata-rata = {silhouette_avg:.4f}')

    ax.set_title(f'Silhouette Analysis Plot\n(K={n_clusters} cluster)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster Label', fontsize=12)
    ax.set_xlim([-0.5, 1.0])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2, axis='x')

    _simpan_dan_tampilkan(fig, path_output, "Silhouette Analysis")


# ============================================================
# 9. RGB DISTRIBUTION PER JENIS KOPI [BARU]
# ============================================================
def plot_rgb_distribution(list_hasil_prep, list_label_asli, path_output=None):
    """
    Memplot distribusi histogram channel R, G, B per jenis kopi
    (arabika, liberika, robusta) untuk membandingkan profil warna.
    """
    if path_output is None:
        path_output = config.RGB_DISTRIBUTION_PATH

    unique_types = sorted(set(list_label_asli))
    colors_map = {'arabika': '#e74c3c', 'liberika': '#2ecc71', 'robusta': '#3498db'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    channel_names = ['R (Red)', 'G (Green)', 'B (Blue)']

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        for tipe in unique_types:
            # Kumpulkan data channel dari semua gambar tipe ini
            all_pixels = []
            for i, (hasil_prep, label) in enumerate(zip(list_hasil_prep, list_label_asli)):
                if label == tipe and hasil_prep is not None:
                    rgb = hasil_prep['rgb_norm']
                    all_pixels.extend(rgb[:, :, ch_idx].flatten().tolist())

            # Sampling agar tidak terlalu banyak data
            if len(all_pixels) > 50000:
                np.random.seed(config.RANDOM_STATE)
                all_pixels = np.random.choice(all_pixels, 50000, replace=False)

            color = colors_map.get(tipe.lower(), 'gray')
            ax.hist(all_pixels, bins=50, alpha=0.5, label=tipe, color=color, density=True)

        ax.set_title(f'Distribusi Channel {ch_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Nilai Piksel (0-1)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Perbandingan Distribusi Warna per Jenis Biji Kopi',
                 fontsize=14, fontweight='bold', y=1.02)

    _simpan_dan_tampilkan(fig, path_output, "RGB Distribution per Type")


# ============================================================
# 10. CLUSTER SIZE BAR CHART [BARU]
# ============================================================
def _ambil_gambar_representatif(cluster_id, df_labeled, images_dir, ukuran=(70, 70)):
    """
    Mengambil satu gambar biji kopi asli yang paling representatif
    (paling mendekati mean RGB cluster) untuk ditampilkan sebagai thumbnail.

    Parameters
    ----------
    cluster_id : int
        ID cluster yang dicari gambarnya.
    df_labeled : pd.DataFrame
        DataFrame labeled_dataset.csv dengan kolom 'nama_file' & 'cluster_label'.
    images_dir : str
        Path ke folder coffee_images (berisi subfolder arabika/robusta/liberika).
    ukuran : tuple
        Ukuran resize thumbnail (lebar, tinggi) dalam piksel.

    Returns
    -------
    np.ndarray atau None
        Array gambar RGB shape (H, W, 3), atau None jika tidak ditemukan.
    """
    def _cari_path(nama_file):
        for subfolder in ['arabika', 'robusta', 'liberika']:
            path = os.path.join(images_dir, subfolder, nama_file)
            if os.path.exists(path):
                return path
        return None

    baris = df_labeled[df_labeled['cluster_label'] == cluster_id]
    if baris.empty:
        return None

    # Ambil sampel kandidat gambar (max 30)
    kandidat = baris.sample(n=min(30, len(baris)), random_state=42)

    imgs_rgb = []
    for nama_file in kandidat['nama_file'].values:
        path = _cari_path(nama_file)
        if path is None:
            continue
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        imgs_rgb.append((nama_file, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

    if not imgs_rgb:
        return None

    # Hitung mean RGB biji kopi (tanpa background putih) per gambar kandidat
    mean_per_gambar = []
    for _, img_rgb in imgs_rgb:
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        brightness = pixels.mean(axis=1)
        mask = brightness < 200
        if mask.sum() < 10:
            mask = brightness < 230
        mean_per_gambar.append(
            pixels[mask].mean(axis=0) if mask.sum() > 0 else pixels.mean(axis=0)
        )

    # Pilih gambar yang mean-nya paling dekat dengan mean keseluruhan cluster
    mean_cluster = np.array(mean_per_gambar).mean(axis=0)
    jarak = [np.linalg.norm(np.array(m) - mean_cluster) for m in mean_per_gambar]
    idx_terbaik = int(np.argmin(jarak))

    _, img_terbaik = imgs_rgb[idx_terbaik]

    # --- Cropping Biji Kopi untuk Menghilangkan Background Putih Berlebih ---
    h_img, w_img, _ = img_terbaik.shape
    # Karena seluruh gambar biji kopi di dataset ini dipotret dengan setup kamera standar 
    # (resolusi seragam dan posisi biji kopi selalu tepat di tengah), cara terbaik dan paling
    # konsisten agar ukuran biji kopi "sama rata" (seragam) di semua cluster adalah dengan
    # memotong area tengah dengan persentase/skala zoom yang persis sama untuk seluruh gambar.
    # Kita potong area tengah sebesar 22% (11% ke atas-bawah-kiri-kanan dari pusat).
    cy, cx = h_img // 2, w_img // 2
    dy, dx = int(h_img * 0.11), int(w_img * 0.11)
    img_cropped = img_terbaik[cy-dy:cy+dy, cx-dx:cx+dx]

    # Resize hasil cropping ke ukuran target
    return cv2.resize(img_cropped, ukuran, interpolation=cv2.INTER_AREA)


def plot_cluster_size(label_cluster, path_output=None):
    """
    Memplot bar chart distribusi ukuran cluster dengan thumbnail gambar
    biji kopi asli di atas setiap bar, serta pie chart proporsi cluster.

    Cara kerja:
      - Setiap bar diwarnai dengan rata-rata RGB gambar asli biji kopi
        di cluster tersebut (tanpa background putih).
      - Di atas setiap bar ditampilkan thumbnail foto biji kopi asli
        yang paling representatif untuk cluster tersebut.
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    if path_output is None:
        path_output = config.CLUSTER_SIZE_PATH

    unique_labels, counts = np.unique(label_cluster, return_counts=True)
    total = counts.sum()
    percentages = (counts / total) * 100
    cluster_labels = [config.get_cluster_name(l) for l in unique_labels]

    # ----------------------------------------------------------------
    # Baca labeled_dataset.csv untuk mapping gambar -> cluster
    # ----------------------------------------------------------------
    df_labeled = pd.read_csv(config.LABELED_DATASET_CSV)
    images_dir = config.COFFEE_IMAGES_DIR

    # ----------------------------------------------------------------
    # Ambil warna rata-rata dari pixel biji kopi (tanpa background)
    # ----------------------------------------------------------------
    print("[INFO] Menghitung warna rata-rata pixel gambar asli per cluster...")
    warna_dari_gambar = ambil_warna_cluster_dari_gambar(
        labeled_csv_path=config.LABELED_DATASET_CSV,
        images_dir=images_dir
    )
    fallback_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    colors = []
    for i, label in enumerate(unique_labels):
        if warna_dari_gambar and warna_dari_gambar.get(label) is not None:
            colors.append(warna_dari_gambar[label])
        else:
            colors.append(tuple(fallback_colors[i]))

    # ----------------------------------------------------------------
    # Ambil gambar representatif per cluster
    # ----------------------------------------------------------------
    print("[INFO] Mengambil gambar representatif per cluster...")
    gambar_per_cluster = {}
    for label in unique_labels:
        img = _ambil_gambar_representatif(
            cluster_id=label,
            df_labeled=df_labeled,
            images_dir=images_dir,
            ukuran=(120, 120)  # Ukuran sedikit lebih besar agar resolusi di bar lebih bagus
        )
        gambar_per_cluster[label] = img
        print(f"  Cluster {label} ({config.get_cluster_name(label)}): "
              f"{'OK' if img is not None else 'tidak ditemukan'}")

    # ----------------------------------------------------------------
    # Layout: bar chart (kiri) + pie chart (kanan)
    # ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.patch.set_facecolor('#FAFAFA')

    # --- Bar chart ---
    x_pos = np.arange(len(cluster_labels))
    bar_width = 0.6

    # Kita buat dummy bars tak terlihat (alpha=0) hanya agar Matplotlib
    # mengonfigurasi axis, limit, dan label secara otomatis.
    dummy_bars = ax1.bar(x_pos, counts, color='none', edgecolor='none', width=bar_width)

    ax1.set_title('Jumlah Data per Wilayah Distribusi',
                  fontsize=13, fontweight='bold', pad=85)
    ax1.set_xlabel('Wilayah Distribusi', fontsize=10)
    ax1.set_ylabel('Jumlah Data', fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cluster_labels, rotation=30, ha='right', fontsize=9)
    ax1.set_ylim(0, counts.max() * 1.5)   # ruang ekstra di atas untuk thumbnail
    ax1.set_xlim(-0.6, len(cluster_labels) - 0.4)  # MEMAKSA rentang X agar semua 8 bar biji kopi terlihat
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_facecolor('#F8F8F8')

    # Menggambar gambar biji kopi asli sebagai tumpukan (stack) di dalam area setiap bar
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        img_arr = gambar_per_cluster.get(label)
        x_center = i
        x_left = x_center - bar_width / 2
        x_right = x_center + bar_width / 2

        if img_arr is not None:
            # Hitung aspect ratio dan proporsi tinggi tumpukan agar seragam
            # Tinggi tumpukan diatur agar satu biji kopi bernilai ~35 unit data di sumbu Y
            h_img, w_img, _ = img_arr.shape
            
            # Tentukan target tinggi per biji kopi di sumbu Y (agar proporsional)
            target_bean_height = 35.0
            
            # Hitung berapa banyak biji kopi yang perlu ditumpuk untuk mencapai 'count'
            num_beans = max(1, int(round(count / target_bean_height)))
            actual_bean_height = count / num_beans  # Tinggi presisi per segmen
            
            # Tumpuk gambar secara vertikal
            for b in range(num_beans):
                y_bottom = b * actual_bean_height
                y_top = (b + 1) * actual_bean_height
                
                ax1.imshow(
                    img_arr,
                    extent=[x_left, x_right, y_bottom, y_top],
                    aspect='auto',  # aspect auto aman karena ukuran segmen [x_left, x_right, y_bottom, y_top] sudah proporsional
                    zorder=2
                )
            
            # Tambahkan outline/border hitam di sekeliling tumpukan agar bentuk bar tetap rapi
            rect = plt.Rectangle(
                (x_left, 0), bar_width, count,
                facecolor='none',
                edgecolor='#333333',
                linewidth=1.5,
                zorder=3
            )
            ax1.add_patch(rect)
        else:
            # Fallback jika tidak ada gambar
            ax1.bar(i, count, color=colors[i], edgecolor='#333333', linewidth=1.5, width=bar_width, zorder=2)

    # Angka di atas bar
    for i, count in enumerate(counts):
        ax1.text(i, count + counts.max() * 0.012,
                 str(count), ha='center', va='bottom',
                 fontweight='bold', fontsize=9, zorder=4)

    # Tempel gambar biji kopi asli berbentuk bulat/kotak kecil sebagai "cap" di atas bar
    for i, (label, count) in enumerate(zip(unique_labels, unique_labels)):
        img_arr = gambar_per_cluster.get(label)
        if img_arr is None:
            continue

        x_center = i
        y_top = counts[i] + counts.max() * 0.07

        offset_img = OffsetImage(img_arr, zoom=0.55)
        offset_img.image.axes = ax1

        ab = AnnotationBbox(
            offset_img,
            xy=(x_center, y_top),
            xycoords='data',
            frameon=True,
            bboxprops=dict(
                boxstyle='round,pad=0.15',
                edgecolor=colors[i],
                linewidth=2.5,
                facecolor='white'
            ),
            pad=0.3,
            zorder=4
        )
        ax1.add_artist(ab)

    # --- Pie chart ---
    wedges, _ = ax2.pie(
        counts,
        labels=None,
        colors=colors,
        startangle=90,
        wedgeprops=dict(linewidth=1.2, edgecolor='white')
    )
    legend_labels = [
        f'{name}  ({p:.1f}%)'
        for name, p in zip(cluster_labels, percentages)
    ]
    ax2.legend(
        wedges, legend_labels,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8.5,
        framealpha=0.9
    )
    ax2.set_title('Proporsi Wilayah', fontsize=13, fontweight='bold')

    fig.suptitle('Distribusi Ukuran Wilayah Distribusi',
                 fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    _simpan_dan_tampilkan(fig, path_output, "Wilayah Size Chart")


# ============================================================
# 11. RADAR CHART (PROFIL FITUR PER CLUSTER) [BARU]
# ============================================================
def plot_radar_chart(df_scaled, label_cluster, max_fitur=10, path_output=None):
    """
    Memplot radar/spider chart yang menunjukkan profil rata-rata fitur
    untuk setiap cluster (fitur terpilih dengan varians tertinggi).
    """
    if path_output is None:
        path_output = config.RADAR_CHART_PATH

    kolom_non_fitur = ['nama_file', 'jenis_kopi', 'cluster_label']
    kolom_fitur = [c for c in df_scaled.columns if c not in kolom_non_fitur]

    # Pilih fitur dengan varians tertinggi
    if len(kolom_fitur) > max_fitur:
        varians = df_scaled[kolom_fitur].var().sort_values(ascending=False)
        kolom_fitur = varians.head(max_fitur).index.tolist()

    unique_labels = sorted(np.unique(label_cluster))
    n_clusters = len(unique_labels)

    # Hitung mean per cluster, lalu normalisasi ke [0,1] untuk radar
    mean_per_cluster = []
    for label in unique_labels:
        mask = label_cluster == label
        cluster_data = df_scaled.loc[mask, kolom_fitur].values
        mean_per_cluster.append(cluster_data.mean(axis=0))

    mean_array = np.array(mean_per_cluster)
    # Normalisasi ke [0,1]
    mins = mean_array.min(axis=0)
    maxs = mean_array.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # hindari divisi nol
    mean_norm = (mean_array - mins) / ranges

    # Buat radar chart
    angles = np.linspace(0, 2 * np.pi, len(kolom_fitur), endpoint=False).tolist()
    angles += angles[:1]  # tutup polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for idx, (label, color) in enumerate(zip(unique_labels, colors)):
        values = mean_norm[idx].tolist()
        values += values[:1]
        cluster_name = config.get_cluster_name(label)
        ax.plot(angles, values, linewidth=2, label=cluster_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    # Singkat nama fitur jika terlalu panjang
    short_names = [n.replace('hist_', 'h_').replace('mean_', 'm_').replace('std_', 's_')
                   for n in kolom_fitur]
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_title(f'Radar Chart - Profil Fitur per Wilayah Distribusi\n({len(kolom_fitur)} fitur teratas)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    _simpan_dan_tampilkan(fig, path_output, "Radar Chart")


# ============================================================
# 12. PAIR PLOT (SCATTER MATRIX) [BARU]
# ============================================================
def plot_pair_plot(df_scaled, label_cluster, max_fitur=5, path_output=None):
    """
    Memplot pair plot (scatter matrix) dari fitur teratas, diwarnai per cluster.
    Menunjukkan hubungan pairwise antar fitur.
    """
    if path_output is None:
        path_output = config.PAIR_PLOT_PATH

    kolom_non_fitur = ['nama_file', 'jenis_kopi', 'cluster_label']
    kolom_fitur = [c for c in df_scaled.columns if c not in kolom_non_fitur]

    # Pilih fitur dengan varians tertinggi
    if len(kolom_fitur) > max_fitur:
        varians = df_scaled[kolom_fitur].var().sort_values(ascending=False)
        kolom_fitur = varians.head(max_fitur).index.tolist()

    df_plot = df_scaled[kolom_fitur].copy()
    df_plot['cluster'] = [config.get_cluster_name(l) for l in label_cluster]

    # Singkat nama kolom
    rename_map = {c: c.replace('hist_', 'h_').replace('mean_', 'm_').replace('std_', 's_')
                  for c in kolom_fitur}
    df_plot = df_plot.rename(columns=rename_map)

    fig = sns.pairplot(df_plot, hue='cluster', palette='Set2',
                       diag_kind='kde', plot_kws={'alpha': 0.5, 's': 15},
                       diag_kws={'alpha': 0.7})
    fig.figure.suptitle('Pair Plot - Scatter Matrix Fitur per Wilayah Distribusi',
                        fontsize=14, fontweight='bold', y=1.02)

    _simpan_dan_tampilkan(fig.figure, path_output, "Pair Plot")


# ============================================================
# 13. METRIC COMPARISON (EVALUASI NAIVE BAYES) [BARU]
# ============================================================
def plot_metric_comparison(hasil_eval_nb, path_output=None):
    """
    Memplot bar chart perbandingan metrik evaluasi Naive Bayes:
    Akurasi, Presisi, Recall, F1-Score.
    """
    if path_output is None:
        path_output = config.METRIC_COMPARISON_PATH

    metrik = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
    nilai = [
        hasil_eval_nb['akurasi'],
        hasil_eval_nb['presisi'],
        hasil_eval_nb['recall'],
        hasil_eval_nb['f1_score'],
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(metrik, nilai, color=colors, edgecolor='black', linewidth=0.8, width=0.6)

    # Tambahkan nilai di atas bar
    for bar, val in zip(bars, nilai):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}\n({val*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylim(0, 1.15)
    ax.set_title('Perbandingan Metrik Evaluasi Naive Bayes\n(Klasifikasi Wilayah Distribusi)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Nilai Metrik', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    _simpan_dan_tampilkan(fig, path_output, "Metric Comparison")


# ============================================================
# 14. PER-CLASS METRICS BAR CHART (BARU - Klasifikasi)
# ============================================================
def plot_per_class_metrics(y_test=None, y_pred=None, hasil_eval_nb=None,
                           label_names=None, path_output=None):
    """
    Memplot grouped bar chart yang menampilkan Precision, Recall, dan F1-Score
    untuk setiap kelas hasil klasifikasi Naive Bayes.

    Bisa dipanggil dengan:
    - hasil_eval_nb (hasil dari evaluasi_naive_bayes)
    - atau langsung y_test + y_pred + label_names
    """
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support

    if path_output is None:
        path_output = config.CLASSIFICATION_PER_CLASS_PATH

    # Ambil data per-class
    per_class_data = None
    final_labels = None

    if hasil_eval_nb is not None and 'per_class' in hasil_eval_nb:
        per_class_data = hasil_eval_nb['per_class']
        final_labels = hasil_eval_nb.get('label_names', list(per_class_data.keys()))
    elif y_test is not None and y_pred is not None:
        # Hitung ulang jika tidak diberikan melalui hasil_eval_nb
        labels_present = sorted(set(y_test.tolist() + y_pred.tolist()))
        p, r, f1, sup = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        per_class_data = {}
        for i, lbl in enumerate(labels_present):
            key = label_names[i] if (label_names and len(label_names) == len(labels_present)) else str(lbl)
            per_class_data[key] = {
                'precision': float(p[i]),
                'recall': float(r[i]),
                'f1': float(f1[i]),
                'support': int(sup[i])
            }
        final_labels = list(per_class_data.keys())
    else:
        print("[ERROR] plot_per_class_metrics: data tidak cukup (butuh hasil_eval_nb atau y_test+y_pred)")
        return

    if not per_class_data:
        print("[ERROR] Tidak ada data per-class untuk diplot.")
        return

    # Siapkan DataFrame untuk seaborn
    records = []
    for lbl in final_labels:
        if lbl in per_class_data:
            records.append({
                'Kelas': str(lbl),
                'Metrik': 'Precision',
                'Nilai': per_class_data[lbl]['precision']
            })
            records.append({
                'Kelas': str(lbl),
                'Metrik': 'Recall',
                'Nilai': per_class_data[lbl]['recall']
            })
            records.append({
                'Kelas': str(lbl),
                'Metrik': 'F1-Score',
                'Nilai': per_class_data[lbl]['f1']
            })

    df_plot = pd.DataFrame(records)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_plot, x='Kelas', y='Nilai', hue='Metrik',
                palette={'Precision': '#3498db', 'Recall': '#2ecc71', 'F1-Score': '#e74c3c'},
                ax=ax, edgecolor='black', linewidth=0.6)

    ax.set_ylim(0, 1.15)
    ax.set_title('Performa Klasifikasi per Wilayah Distribusi (Naive Bayes)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Wilayah Distribusi', fontsize=12)
    ax.set_ylabel('Nilai Metrik', fontsize=12)
    ax.legend(title='Metrik', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Tambahkan nilai kecil di atas bar jika memungkinkan
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)

    _simpan_dan_tampilkan(fig, path_output, "Per-Class Metrics (Classification)")


# ============================================================
# FUNGSI BANTUAN: JALANKAN SEMUA VISUALISASI STATISTIK
# ============================================================
def jalankan_semua_visualisasi_statistik(
    df_scaled, label_cluster, X_fitur, hasil_elbow,
    hasil_eval_nb, list_hasil_prep=None, list_label_asli=None
):
    """
    Menjalankan semua plot statistik tambahan sekaligus.
    Dipanggil di akhir pipeline setelah K-Means dan Naive Bayes selesai.

    Parameter:
        df_scaled (pd.DataFrame): DataFrame fitur scaled.
        label_cluster (np.ndarray): Label cluster per data.
        X_fitur (np.ndarray): Array fitur numerik.
        hasil_elbow (dict): Hasil Elbow Method.
        hasil_eval_nb (dict): Hasil evaluasi Naive Bayes.
        list_hasil_prep (list): Hasil preprocessing (untuk RGB plot).
        list_label_asli (list): Label asli per gambar (arabika/liberika/robusta).
    """
    print("\n" + "=" * 60)
    print("  VISUALISASI STATISTIK LENGKAP (Wilayah Distribusi)")
    print("=" * 60)

    # 1. PCA Scatter 2D
    print("\n[VIS 1/8] PCA Scatter Plot 2D...")
    plot_pca_scatter(X_fitur, label_cluster)

    # 2. Correlation Heatmap
    print("[VIS 2/8] Correlation Heatmap...")
    plot_correlation_heatmap(df_scaled)

    # 3. Silhouette Analysis
    print("[VIS 3/8] Silhouette Analysis Plot...")
    plot_silhouette_analysis(X_fitur, label_cluster)

    # 4. RGB Distribution
    if list_hasil_prep is not None and list_label_asli is not None:
        print("[VIS 4/8] RGB Distribution per Jenis Kopi...")
        plot_rgb_distribution(list_hasil_prep, list_label_asli)
    else:
        print("[VIS 4/8] RGB Distribution - dilewati (data tidak tersedia)")

    # 5. Wilayah Size
    print("[VIS 5/8] Ukuran Wilayah Distribusi...")
    plot_cluster_size(label_cluster)

    # 6. Radar Chart
    print("[VIS 6/8] Radar Chart...")
    plot_radar_chart(df_scaled, label_cluster)

    # 7. Pair Plot
    print("[VIS 7/8] Pair Plot (Scatter Matrix)...")
    plot_pair_plot(df_scaled, label_cluster)

    # 8. Metric Comparison
    print("[VIS 8/8] Metric Comparison (Naive Bayes)...")
    plot_metric_comparison(hasil_eval_nb)

    # 9. Per-Class Metrics (BARU)
    print("[VIS 9/9] Per-Class Classification Metrics...")
    plot_per_class_metrics(hasil_eval_nb=hasil_eval_nb)

    print("\n[OK] Semua visualisasi statistik selesai!")


# ============================================================
# TEST MODUL (jalankan langsung)
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: visualization.py")
    print("=" * 50)

    # Test Elbow Curve
    k_vals = [2, 3, 4, 5, 6]
    inertia_vals = [5000, 3000, 2000, 1500, 1200]
    plot_elbow_curve(k_vals, inertia_vals, k_optimal=3)

    # Test Confusion Matrix
    y_t = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_p = np.array([0, 0, 1, 1, 2, 1, 0, 2, 2])
    plot_confusion_matrix(y_t, y_p, label_names=['A', 'B', 'C'])

    print("=" * 50)
    print("Test visualization selesai.")
