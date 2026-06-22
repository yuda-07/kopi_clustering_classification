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
- Cluster Size Bar Chart (ukuran cluster)
- Radar Chart (profil fitur per cluster)
- Pair Plot (scatter matrix fitur teratas)
- Metric Comparison (perbandingan metrik evaluasi)

Semua plot: TAMPILKAN POPUP + SIMPAN PNG
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, silhouette_samples
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Gunakan backend yang mendukung popup (TkAgg default)
matplotlib.use('TkAgg')


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
def plot_confusion_matrix(y_test, y_pred, label_names=None, path_output=None):
    """Memplot Confusion Matrix sebagai heatmap."""
    if path_output is None:
        path_output = config.CONFUSION_MATRIX_PATH

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title('Confusion Matrix - Naive Bayes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prediksi', fontsize=12)
    ax.set_ylabel('Sebenarnya', fontsize=12)

    _simpan_dan_tampilkan(fig, path_output, "Confusion Matrix")


# ============================================================
# 4. DISTRIBUSI FITUR PER CLUSTER (BOXPLOT)
# ============================================================
def plot_distribusi_cluster(df_scaled, label_cluster, fitur_pilih=None, path_output=None):
    """Memplot distribusi fitur per cluster menggunakan boxplot."""
    if path_output is None:
        path_output = config.FEATURE_DISTRIBUTION_PATH

    kolom_fitur = [c for c in df_scaled.columns if c != 'nama_file']
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

    fig.suptitle('Distribusi Fitur per Cluster', fontsize=14, fontweight='bold', y=1.02)

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
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=f'Cluster {label}',
                   alpha=0.6, s=30, edgecolors='w', linewidth=0.3)

    # Tandai centroid di ruang PCA
    for label, color in zip(unique_labels, colors):
        mask = label_cluster == label
        cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
        ax.scatter(cx, cy, c=[color], marker='X', s=200, edgecolors='black', linewidth=2)

    ax.set_title(f'PCA Scatter Plot 2D - Visualisasi Cluster\n'
                 f'(PC1={var1:.1f}% variance, PC2={var2:.1f}% variance)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Principal Component 1 ({var1:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({var2:.1f}%)', fontsize=12)
    ax.legend(loc='best', fontsize=9)
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

    kolom_fitur = [c for c in df_scaled.columns if c != 'nama_file']

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
def plot_cluster_size(label_cluster, path_output=None):
    """
    Memplot bar chart + persentase ukuran setiap cluster.
    """
    if path_output is None:
        path_output = config.CLUSTER_SIZE_PATH

    unique_labels, counts = np.unique(label_cluster, return_counts=True)
    total = counts.sum()
    percentages = (counts / total) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    bars = ax1.bar([f'Cluster {l}' for l in unique_labels], counts,
                   color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_title('Jumlah Data per Cluster', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Jumlah Data')

    # Tambahkan angka di atas bar
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontweight='bold')

    # Pie chart
    ax2.pie(counts, labels=[f'Cluster {l}\n({p:.1f}%)' for l, p in zip(unique_labels, percentages)],
            colors=colors, autopct='', startangle=90, textprops={'fontsize': 9})
    ax2.set_title('Proporsi Cluster', fontsize=13, fontweight='bold')

    fig.suptitle('Distribusi Ukuran Cluster', fontsize=14, fontweight='bold', y=1.02)

    _simpan_dan_tampilkan(fig, path_output, "Cluster Size Chart")


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

    kolom_fitur = [c for c in df_scaled.columns if c != 'nama_file']

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
        ax.plot(angles, values, linewidth=2, label=f'Cluster {label}', color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    # Singkat nama fitur jika terlalu panjang
    short_names = [n.replace('hist_', 'h_').replace('mean_', 'm_').replace('std_', 's_')
                   for n in kolom_fitur]
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_title(f'Radar Chart - Profil Fitur per Cluster\n({len(kolom_fitur)} fitur teratas)',
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

    kolom_fitur = [c for c in df_scaled.columns if c != 'nama_file']

    # Pilih fitur dengan varians tertinggi
    if len(kolom_fitur) > max_fitur:
        varians = df_scaled[kolom_fitur].var().sort_values(ascending=False)
        kolom_fitur = varians.head(max_fitur).index.tolist()

    df_plot = df_scaled[kolom_fitur].copy()
    df_plot['cluster'] = [f'C{l}' for l in label_cluster]

    # Singkat nama kolom
    rename_map = {c: c.replace('hist_', 'h_').replace('mean_', 'm_').replace('std_', 's_')
                  for c in kolom_fitur}
    df_plot = df_plot.rename(columns=rename_map)

    fig = sns.pairplot(df_plot, hue='cluster', palette='Set2',
                       diag_kind='kde', plot_kws={'alpha': 0.5, 's': 15},
                       diag_kws={'alpha': 0.7})
    fig.figure.suptitle('Pair Plot - Scatter Matrix Fitur Teratas',
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
    ax.set_title('Perbandingan Metrik Evaluasi Naive Bayes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Nilai Metrik', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    _simpan_dan_tampilkan(fig, path_output, "Metric Comparison")


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
    print("  VISUALISASI STATISTIK LENGKAP")
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

    # 5. Cluster Size
    print("[VIS 5/8] Cluster Size Bar Chart...")
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
