"""
data_loader.py - Modul Pemuat Data
====================================
Berfungsi membaca gambar biji kopi dari folder per grade
dan membaca file CSV metadata.

Return: list gambar (numpy array) + label grade + metadata DataFrame
"""

import os
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def muat_gambar_dari_folder(folder_path):
    """
    Membaca semua gambar dari sebuah folder dan mengembalikannya
    sebagai list numpy array beserta nama file.

    Parameter:
        folder_path (str): Path ke folder yang berisi gambar.

    Return:
        tuple: (list_gambar: list[np.ndarray], list_nama_file: list[str])
    """
    list_gambar = []
    list_nama_file = []

    try:
        # Cek apakah folder ada
        if not os.path.exists(folder_path):
            print(f"[ERROR] Folder tidak ditemukan: {folder_path}")
            return list_gambar, list_nama_file

        # Iterasi semua file di folder
        for nama_file in sorted(os.listdir(folder_path)):
            # Cek ekstensi file yang didukung
            ekstensi = os.path.splitext(nama_file)[1].lower()
            if ekstensi in config.IMAGE_EXTENSIONS:
                path_gambar = os.path.join(folder_path, nama_file)
                # Baca gambar menggunakan OpenCV (format BGR)
                gambar = cv2.imread(path_gambar)

                if gambar is not None:
                    list_gambar.append(gambar)
                    list_nama_file.append(nama_file)
                else:
                    print(f"[ERROR] Gagal membaca gambar: {path_gambar}")

        print(f"[OK] Berhasil memuat {len(list_gambar)} gambar dari: {folder_path}")

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat memuat gambar: {e}")

    return list_gambar, list_nama_file


def muat_semua_grade():
    """
    Membaca gambar dari semua subfolder grade (Grade_A, Grade_B, Grade_C)
    di dalam folder coffee_images.

    Return:
        tuple: (list_gambar: list[np.ndarray], list_label: list[str], list_nama: list[str])
    """
    semua_gambar = []
    semua_label = []
    semua_nama = []

    print("[INFO] Memulai pemuatan gambar dari semua grade...")

    for grade in config.GRADE_LABELS:
        folder_grade = os.path.join(config.COFFEE_IMAGES_DIR, grade)
        gambar, nama_file = muat_gambar_dari_folder(folder_grade)

        if len(gambar) > 0:
            # Tambahkan label grade untuk setiap gambar
            label_grade = [grade] * len(gambar)
            semua_gambar.extend(gambar)
            semua_label.extend(label_grade)
            semua_nama.extend(nama_file)
        else:
            print(f"[INFO] Tidak ada gambar di folder grade: {grade}")

    total = len(semua_gambar)
    print(f"[OK] Total gambar berhasil dimuat: {total}")
    for grade in config.GRADE_LABELS:
        jumlah = semua_label.count(grade)
        print(f"     - {grade}: {jumlah} gambar")

    return semua_gambar, semua_label, semua_nama


def muat_csv_kualitas():
    """
    Membaca file CSV metadata kualitas kopi dari Kaggle.

    Return:
        pd.DataFrame atau None: DataFrame jika berhasil, None jika gagal.
    """
    csv_path = config.COFFEE_QUALITY_CSV

    try:
        if not os.path.exists(csv_path):
            print(f"[ERROR] File CSV tidak ditemukan: {csv_path}")
            return None

        df = pd.read_csv(csv_path)
        print(f"[OK] CSV kualitas kopi berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df

    except Exception as e:
        print(f"[ERROR] Gagal membaca CSV kualitas: {e}")
        return None


def muat_csv_wilayah():
    """
    Membaca file CSV metadata wilayah distribusi.

    Return:
        pd.DataFrame atau None: DataFrame jika berhasil, None jika gagal.
    """
    csv_path = config.DISTRIBUTION_REGION_CSV

    try:
        if not os.path.exists(csv_path):
            print(f"[ERROR] File CSV tidak ditemukan: {csv_path}")
            return None

        df = pd.read_csv(csv_path)
        print(f"[OK] CSV wilayah distribusi berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df

    except Exception as e:
        print(f"[ERROR] Gagal membaca CSV wilayah: {e}")
        return None


def muat_fitur_terekstrak():
    """
    Membaca file CSV hasil ekstraksi fitur (features_extracted.csv).

    Return:
        pd.DataFrame atau None: DataFrame fitur jika berhasil, None jika gagal.
    """
    csv_path = config.FEATURES_EXTRACTED_CSV

    try:
        if not os.path.exists(csv_path):
            print(f"[ERROR] File fitur tidak ditemukan: {csv_path}")
            print("[INFO] Jalankan ekstraksi fitur terlebih dahulu.")
            return None

        df = pd.read_csv(csv_path)
        print(f"[OK] Fitur terekstrak berhasil dimuat: {df.shape[0]} data, {df.shape[1]} kolom")
        return df

    except Exception as e:
        print(f"[ERROR] Gagal membaca fitur terekstrak: {e}")
        return None


# ============================================================
# TEST MODUL (jalankan langsung)
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: data_loader.py")
    print("=" * 50)

    # Muat gambar dari semua grade
    gambar, label, nama = muat_semua_grade()

    # Muat CSV metadata
    df_kualitas = muat_csv_kualitas()
    df_wilayah = muat_csv_wilayah()

    print("=" * 50)
    print("Test data_loader selesai.")
