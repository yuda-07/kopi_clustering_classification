"""
image_preprocessing.py - Modul Preprocessing Gambar
=====================================================
Melakukan preprocessing pada gambar biji kopi:
- Resize ke 128x128 piksel
- Konversi BGR → RGB dan BGR → HSV
- Gaussian Blur untuk noise removal
- Normalisasi nilai piksel (0–1)
"""

import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def resize_gambar(gambar, ukuran=None):
    """
    Resize gambar ke ukuran yang ditentukan.

    Parameter:
        gambar (np.ndarray): Gambar input (BGR).
        ukuran (int): Ukuran sisi (default dari config.IMG_SIZE).

    Return:
        np.ndarray: Gambar yang sudah di-resize.
    """
    if ukuran is None:
        ukuran = config.IMG_SIZE
    return cv2.resize(gambar, (ukuran, ukuran), interpolation=cv2.INTER_AREA)


def konversi_ke_rgb(gambar_bgr):
    """
    Mengkonversi gambar dari format BGR (OpenCV default) ke RGB.

    Parameter:
        gambar_bgr (np.ndarray): Gambar format BGR.

    Return:
        np.ndarray: Gambar format RGB.
    """
    return cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)


def konversi_ke_hsv(gambar_bgr):
    """
    Mengkonversi gambar dari format BGR ke HSV.

    Parameter:
        gambar_bgr (np.ndarray): Gambar format BGR.

    Return:
        np.ndarray: Gambar format HSV.
    """
    return cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2HSV)


def konversi_ke_grayscale(gambar_bgr):
    """
    Mengkonversi gambar dari format BGR ke Grayscale.

    Parameter:
        gambar_bgr (np.ndarray): Gambar format BGR.

    Return:
        np.ndarray: Gambar grayscale (2D array).
    """
    return cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)


def hapus_noise(gambar, kernel_size=None):
    """
    Menghilangkan noise menggunakan Gaussian Blur.

    Parameter:
        gambar (np.ndarray): Gambar input.
        kernel_size (tuple): Ukuran kernel blur (default dari config).

    Return:
        np.ndarray: Gambar yang sudah di-blur.
    """
    if kernel_size is None:
        kernel_size = config.BLUR_KERNEL_SIZE
    return cv2.GaussianBlur(gambar, kernel_size, 0)


def normalisasi_piksel(gambar):
    """
    Normalisasi nilai piksel ke rentang [0, 1] (float32).

    Parameter:
        gambar (np.ndarray): Gambar input (uint8, 0-255).

    Return:
        np.ndarray: Gambar dengan nilai piksel 0.0 - 1.0 (float32).
    """
    return gambar.astype(np.float32) / 255.0


def buat_mask_objek(gambar_bgr):
    """
    Membuat mask biner untuk dataset training (latar belakang putih/seragam).
    Menggunakan Otsu Thresholding — terbukti akurat untuk dataset standar 640x640.
    Foreground (biji kopi) = 255 (uint8), Background = 0.
    """
    gray = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape
    area_total = h * w

    # Thresholding Otsu untuk latar belakang putih
    _, thresh_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_norm = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts_inv, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_norm, _ = cv2.findContours(thresh_norm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    for cnts in [cnts_inv, cnts_norm]:
        valid = [c for c in cnts if (area_total * 0.95) > cv2.contourArea(c) > (area_total * 0.01)]
        if valid:
            c = max(valid, key=cv2.contourArea)
            if best_cnt is None or cv2.contourArea(c) > cv2.contourArea(best_cnt):
                best_cnt = c

    mask = np.zeros((h, w), dtype=np.uint8)
    if best_cnt is not None:
        cv2.drawContours(mask, [best_cnt], -1, 255, -1)
    else:
        margin_h, margin_w = int(h * 0.05), int(w * 0.05)
        mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255

    return mask


def buat_mask_realworld(gambar_bgr):
    """
    Membuat mask untuk foto real-world dari kamera HP (latar belakang lantai/meja bervariasi).
    Menggunakan filter HSV Saturation untuk mengisolasi warna cokelat biji kopi sangrai
    dari latar belakang netral (lantai keramik, bayangan, dll).
    """
    h, w = gambar_bgr.shape[:2]
    area_total = h * w

    hsv = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Filter: biji kopi cokelat memiliki Saturation > 20 dan Value < 190
    mask_kopi = ((s_channel > 20) & (v_channel < 190)).astype(np.uint8) * 255

    # Adaptive threshold untuk menangkap biji sangat gelap
    gray = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_local = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 5)
    dark_beans = (v_channel < 140) & (s_channel > 15)
    combined = cv2.bitwise_or(mask_kopi, thresh_local & dark_beans.astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)

    valid_cnts = [c for c in cnts if (area_total * 0.20) > cv2.contourArea(c) > (area_total * 0.0008)]
    if valid_cnts:
        for c in valid_cnts:
            cv2.drawContours(mask, [c], -1, 255, -1)

    # Fallback jika mask terlalu kecil
    if np.sum(mask > 0) < (area_total * 0.02):
        margin_h, margin_w = int(h * 0.1), int(w * 0.1)
        mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255

    return mask


def preprocessing_gambar(gambar_bgr):
    """
    Pipeline preprocessing lengkap untuk satu gambar.
    Melakukan: resize → noise removal → konversi RGB → konversi HSV → pembuatan mask → normalisasi.

    Parameter:
        gambar_bgr (np.ndarray): Gambar input format BGR dari OpenCV.

    Return:
        dict: Dictionary berisi gambar hasil preprocessing & mask objek.
    """
    try:
        # Tahap 1: Resize gambar ke ukuran standar
        gambar_resized = resize_gambar(gambar_bgr)

        # Tahap 2: Hilangkan noise dengan Gaussian Blur
        gambar_blur = hapus_noise(gambar_resized)

        # Tahap 3: Konversi ke berbagai ruang warna
        gambar_rgb = konversi_ke_rgb(gambar_blur)
        gambar_hsv = konversi_ke_hsv(gambar_resized)
        gambar_gray = konversi_ke_grayscale(gambar_blur)

        # Tahap 4: Buat mask objek biji kopi (Background removal)
        mask_objek = buat_mask_objek(gambar_resized)

        # Tahap 5: Normalisasi piksel ke rentang [0, 1]
        rgb_norm = normalisasi_piksel(gambar_rgb)
        hsv_norm = normalisasi_piksel(gambar_hsv)
        gray_norm = normalisasi_piksel(gambar_gray)

        return {
            'rgb_norm': rgb_norm,
            'hsv_norm': hsv_norm,
            'gray_norm': gray_norm,
            'rgb': gambar_rgb,
            'hsv': gambar_hsv,
            'gray': gambar_gray,
            'mask': mask_objek,
        }

    except Exception as e:
        print(f"[ERROR] Gagal preprocessing gambar: {e}")
        return None


def preprocessing_batch(list_gambar):
    """
    Melakukan preprocessing pada batch (list) gambar sekaligus.

    Parameter:
        list_gambar (list): List gambar BGR (numpy array).

    Return:
        list: List dictionary hasil preprocessing per gambar.
    """
    hasil_batch = []
    total = len(list_gambar)

    print(f"[INFO] Memulai preprocessing {total} gambar...")

    for i, gambar in enumerate(list_gambar):
        hasil = preprocessing_gambar(gambar)
        if hasil is not None:
            hasil_batch.append(hasil)
        else:
            print(f"[ERROR] Preprocessing gambar ke-{i+1} gagal, dilewati.")

        # Progress indicator setiap 50 gambar
        if (i + 1) % 50 == 0:
            print(f"[INFO] Progress: {i+1}/{total} gambar diproses...")

    print(f"[OK] Preprocessing selesai: {len(hasil_batch)}/{total} gambar berhasil.")
    return hasil_batch


# ============================================================
# TEST MODUL (jalankan langsung)
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: image_preprocessing.py")
    print("=" * 50)

    # Buat gambar dummy untuk testing
    dummy_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    hasil = preprocessing_gambar(dummy_img)
    if hasil:
        print(f"  rgb_norm shape : {hasil['rgb_norm'].shape}, dtype: {hasil['rgb_norm'].dtype}")
        print(f"  hsv_norm shape : {hasil['hsv_norm'].shape}, dtype: {hasil['hsv_norm'].dtype}")
        print(f"  gray_norm shape: {hasil['gray_norm'].shape}, dtype: {hasil['gray_norm'].dtype}")
        print(f"  range nilai    : [{hasil['rgb_norm'].min():.2f}, {hasil['rgb_norm'].max():.2f}]")

    print("=" * 50)
    print("Test image_preprocessing selesai.")
