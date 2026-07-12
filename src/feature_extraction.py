"""
feature_extraction.py - Modul Ekstraksi Fitur
===============================================
Mengekstrak fitur dari gambar biji kopi yang sudah dipreprocessing:
- Statistik warna: mean & std tiap channel R, G, B dan H, S, V
- Histogram warna (32 bin per channel)
- Fitur tekstur GLCM: kontras, energi, homogenitas, dissimilarity

Return: DataFrame fitur per gambar
"""

import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def ekstraksi_fitur_warna(rgb_norm, hsv_norm):
    """
    Menghitung mean dan standar deviasi untuk setiap channel warna RGB dan HSV.

    Parameter:
        rgb_norm (np.ndarray): Gambar RGB ternormalisasi (H, W, 3).
        hsv_norm (np.ndarray): Gambar HSV ternormalisasi (H, W, 3).

    Return:
        dict: Fitur statistik warna (12 fitur total).
    """
    fitur = {}

    # Fitur RGB: mean dan std per channel
    for i, channel in enumerate(['R', 'G', 'B']):
        fitur[f'mean_{channel}'] = np.mean(rgb_norm[:, :, i])
        fitur[f'std_{channel}'] = np.std(rgb_norm[:, :, i])

    # Fitur HSV: mean dan std per channel
    for i, channel in enumerate(['H', 'S', 'V']):
        fitur[f'mean_{channel}'] = np.mean(hsv_norm[:, :, i])
        fitur[f'std_{channel}'] = np.std(hsv_norm[:, :, i])

    return fitur


def ekstraksi_histogram(rgb_norm, bins=None):
    """
    Menghitung histogram warna untuk setiap channel RGB.

    Parameter:
        rgb_norm (np.ndarray): Gambar RGB ternormalisasi (H, W, 3).
        bins (int): Jumlah bin histogram (default dari config).

    Return:
        dict: Fitur histogram per channel (32 bin x 3 channel = 96 fitur).
    """
    if bins is None:
        bins = config.HISTOGRAM_BINS

    fitur = {}

    for i, channel in enumerate(['R', 'G', 'B']):
        # Hitung histogram untuk channel ini
        hist, _ = np.histogram(rgb_norm[:, :, i], bins=bins, range=(0, 1))
        # Normalisasi histogram agar jumlah total = 1
        hist = hist.astype(np.float32) / hist.sum()

        # Simpan setiap bin sebagai fitur terpisah
        for j in range(bins):
            fitur[f'hist_{channel}_bin{j:02d}'] = hist[j]

    return fitur


def ekstraksi_fitur_tekstur(gray_uint8):
    """
    Menghitung fitur tekstur menggunakan GLCM (Gray-Level Co-occurrence Matrix).
    Fitur yang dihitung: kontras, energi (ASM), homogenitas, dissimilarity.

    Parameter:
        gray_uint8 (np.ndarray): Gambar grayscale uint8 (0-255), 2D array.

    Return:
        dict: Fitur tekstur GLCM (4 fitur).
    """
    fitur = {}

    try:
        # Konversi ke uint8 jika belum (harus 8-bit untuk GLCM)
        if gray_uint8.dtype != np.uint8:
            gray_uint8 = (gray_uint8 * 255).astype(np.uint8)

        # Hitung GLCM dengan jarak=1, sudut=0 (horizontal)
        # Levels=256 untuk citra 8-bit
        glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256,
                            symmetric=True, normed=True)

        # Ekstrak properti tekstur
        fitur['kontras'] = graycoprops(glcm, 'contrast')[0, 0]
        fitur['energi'] = graycoprops(glcm, 'energy')[0, 0]
        fitur['homogenitas'] = graycoprops(glcm, 'homogeneity')[0, 0]
        fitur['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]

    except Exception as e:
        print(f"[ERROR] Gagal menghitung GLCM: {e}")
        fitur['kontras'] = 0.0
        fitur['energi'] = 0.0
        fitur['homogenitas'] = 0.0
        fitur['dissimilarity'] = 0.0

    return fitur


def ekstraksi_fitur_gambar(hasil_preprocessing, nama_file=""):
    """
    Fungsi utama: mengekstrak semua fitur dari satu gambar yang sudah dipreprocessing.

    Parameter:
        hasil_preprocessing (dict): Hasil dari image_preprocessing.preprocessing_gambar().
        nama_file (str): Nama file gambar (untuk identifikasi).

    Return:
        dict: Semua fitur yang diekstrak (112+ fitur total).
    """
    try:
        # Ambil komponen gambar dari hasil preprocessing
        rgb_norm = hasil_preprocessing['rgb_norm']
        hsv_norm = hasil_preprocessing['hsv_norm']
        gray = hasil_preprocessing['gray']  # uint8 untuk GLCM

        # Kumpulkan semua fitur
        semua_fitur = {'nama_file': nama_file}

        # 1. Fitur statistik warna (12 fitur)
        fitur_warna = ekstraksi_fitur_warna(rgb_norm, hsv_norm)
        semua_fitur.update(fitur_warna)

        # 2. Fitur histogram warna (96 fitur)
        fitur_hist = ekstraksi_histogram(rgb_norm)
        semua_fitur.update(fitur_hist)

        # 3. Fitur tekstur GLCM (4 fitur)
        fitur_tekstur = ekstraksi_fitur_tekstur(gray)
        semua_fitur.update(fitur_tekstur)

        return semua_fitur

    except Exception as e:
        print(f"[ERROR] Gagal ekstraksi fitur untuk {nama_file}: {e}")
        return None


def ekstraksi_batch(list_hasil_preprocessing, list_nama_file, list_label=None):
    """
    Mengekstrak fitur dari batch gambar yang sudah dipreprocessing.

    Parameter:
        list_hasil_preprocessing (list): List hasil preprocessing per gambar.
        list_nama_file (list): List nama file per gambar.
        list_label (list, opsional): List label jenis kopi asli.

    Return:
        pd.DataFrame: DataFrame berisi semua fitur per gambar.
    """
    semua_fitur = []
    total = len(list_hasil_preprocessing)

    print(f"[INFO] Memulai ekstraksi fitur untuk {total} gambar...")

    for i in range(total):
        hasil_prep = list_hasil_preprocessing[i]
        nama = list_nama_file[i]
        label = list_label[i] if list_label is not None else None

        fitur = ekstraksi_fitur_gambar(hasil_prep, nama_file=nama)
        if fitur is not None:
            if label is not None:
                fitur['jenis_kopi'] = label
            semua_fitur.append(fitur)
        else:
            print(f"[ERROR] Ekstraksi fitur gambar ke-{i+1} gagal, dilewati.")

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"[INFO] Progress ekstraksi: {i+1}/{total} gambar...")

    df_fitur = pd.DataFrame(semua_fitur)

    # Letakkan jenis_kopi di depan (setelah nama_file) jika ada
    if 'jenis_kopi' in df_fitur.columns:
        cols = list(df_fitur.columns)
        cols.remove('jenis_kopi')
        cols.insert(1, 'jenis_kopi')
        df_fitur = df_fitur[cols]

    print(f"[OK] Ekstraksi fitur selesai: {df_fitur.shape[0]} data, {df_fitur.shape[1]} kolom fitur.")

    return df_fitur


def simpan_fitur(df_fitur, path_output=None):
    """
    Menyimpan DataFrame fitur ke file CSV.

    Parameter:
        df_fitur (pd.DataFrame): DataFrame fitur.
        path_output (str): Path file output (default dari config).
    """
    if path_output is None:
        path_output = config.FEATURES_EXTRACTED_CSV

    try:
        df_fitur.to_csv(path_output, index=False)
        print(f"[OK] Fitur berhasil disimpan ke: {path_output}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan fitur: {e}")


# ============================================================
# TEST MODUL (jalankan langsung)
# ============================================================
if __name__ == "__main__":
    from image_preprocessing import preprocessing_gambar

    print("=" * 50)
    print("TEST: feature_extraction.py")
    print("=" * 50)

    # Buat gambar dummy
    dummy_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    hasil_prep = preprocessing_gambar(dummy_img)

    if hasil_prep:
        fitur = ekstraksi_fitur_gambar(hasil_prep, nama_file="dummy_test.jpg")
        if fitur:
            # Hitung jumlah fitur
            jumlah_fitur = len([k for k in fitur.keys() if k != 'nama_file'])
            print(f"  Jumlah fitur diekstrak: {jumlah_fitur}")
            print(f"  Contoh fitur warna : mean_R={fitur['mean_R']:.4f}, std_G={fitur['std_G']:.4f}")
            print(f"  Contoh fitur tekstur: kontras={fitur['kontras']:.4f}, energi={fitur['energi']:.4f}")

    print("=" * 50)
    print("Test feature_extraction selesai.")
