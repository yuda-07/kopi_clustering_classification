"""
cek_gambar.py - Script Cepat Cek Prediksi Gambar Kopi (Multi-Gambar)
=====================================================================
Jalankan: python cek_gambar.py

Fitur:
- Mendukung pengecekan BANYAK gambar sekaligus
- Setiap gambar menghasilkan output PNG dengan bounding box merah + label prediksi
- Di akhir dibuat gambar GABUNGAN (grid) semua hasil sekaligus
"""

import os
import sys
import cv2
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src import image_preprocessing, feature_extraction

# ============================================================
# KONFIGURASI - tambahkan / ganti path gambar di sini
# ============================================================
DAFTAR_GAMBAR = [
    r"d:\AI\kopi_clustering_classification\arabika_1.jpg",
    r"d:\AI\kopi_clustering_classification\robusta_1.jpg",
    r"d:\AI\kopi_clustering_classification\liberika_3.jpg",
]

# Warna bounding box per jenis kopi (BGR format OpenCV)
WARNA_PER_KELAS = {
    "arabika":  (0,  200,  0),    # Hijau
    "robusta":  (0,   0, 255),    # Merah
    "liberika": (255, 165,  0),   # Biru-Oranye
}
WARNA_DEFAULT = (0, 0, 255)


# ============================================================
# FUNGSI HELPER
# ============================================================
def muat_model():
    """Muat scaler, model, dan label encoder dari file .pkl."""
    scaler        = joblib.load(config.VARIETY_SCALER_PATH)
    model         = joblib.load(config.VARIETY_MODEL_PATH)
    label_encoder = joblib.load(config.VARIETY_LABEL_ENCODER_PATH)
    return scaler, model, label_encoder


def deteksi_latar(img_bgr):
    """
    Deteksi otomatis apakah gambar berasal dari dataset terstandar
    (latar putih) atau foto real-world dari HP (latar bervariasi).
    Caranya: cek rata-rata brightness 10% border/pinggir gambar.
    Jika border sangat terang (rata2 > 200), berarti latar putih → dataset.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    margin_h = max(1, int(h * 0.10))
    margin_w = max(1, int(w * 0.10))

    border_top    = gray[:margin_h, :]
    border_bottom = gray[h-margin_h:, :]
    border_left   = gray[:, :margin_w]
    border_right  = gray[:, w-margin_w:]

    border_pixels = np.concatenate([
        border_top.flatten(),
        border_bottom.flatten(),
        border_left.flatten(),
        border_right.flatten()
    ])

    avg_brightness = np.mean(border_pixels)

    if avg_brightness > 200:
        return "dataset"    # Latar putih bersih → gunakan Otsu (sama dengan training)
    else:
        return "realworld"  # Latar bervariasi → gunakan HSV filter


def prediksi_gambar(path_gambar, scaler, model, label_encoder):
    """
    Jalankan full pipeline prediksi untuk satu gambar.
    Auto-deteksi jenis latar belakang untuk memilih mask yang tepat:
    - Latar putih (dataset): pakai buat_mask_objek (sama persis dengan training)
    - Latar bervariasi (foto HP): pakai buat_mask_realworld
    Ini memastikan fitur yang diekstrak konsisten dengan fitur saat training.
    """
    hasil = {"berhasil": False, "path": path_gambar}

    img_bgr = cv2.imread(path_gambar)
    if img_bgr is None:
        print(f"  [ERROR] Gagal membaca: {path_gambar}")
        return hasil

    # Preprocessing standar (resize, blur, normalisasi)
    hasil_prep = image_preprocessing.preprocessing_gambar(img_bgr)
    if hasil_prep is None:
        print(f"  [ERROR] Preprocessing gagal: {os.path.basename(path_gambar)}")
        return hasil

    # Auto-deteksi latar belakang pada gambar yang sudah di-resize
    img_resized = cv2.resize(img_bgr, (config.IMG_SIZE, config.IMG_SIZE))
    jenis_latar = deteksi_latar(img_resized)

    if jenis_latar == "dataset":
        # White background → use SAME mask as training (Otsu)
        mask_final = image_preprocessing.buat_mask_objek(img_resized)
        print(f"  [INFO] Detected background: WHITE (dataset mode) → Otsu mask")
    else:
        # Varied background → use HSV mask for phone photos
        mask_final = image_preprocessing.buat_mask_realworld(img_resized)
        print(f"  [INFO] Detected background: VARIED (real-world mode) → HSV mask")

    hasil_prep['mask'] = mask_final

    # Feature extraction
    fitur_dict = feature_extraction.ekstraksi_fitur_gambar(
        hasil_prep, nama_file=os.path.basename(path_gambar)
    )
    if fitur_dict is None:
        print(f"  [ERROR] Feature extraction failed: {os.path.basename(path_gambar)}")
        return hasil

    # Align feature order with scaler
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)
    elif hasattr(scaler, 'feature_names_'):
        feature_names = list(scaler.feature_names_)
    else:
        feature_names = [k for k in fitur_dict.keys() if k != 'nama_file']

    feature_vector = [fitur_dict.get(name, 0.0) for name in feature_names]

    X_input  = np.array([feature_vector])
    X_scaled = scaler.transform(X_input)

    # Prediction
    pred_class = model.predict(X_scaled)[0]
    pred_probs = model.predict_proba(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    hasil.update({
        "berhasil":   True,
        "pred_label": pred_label,
        "pred_probs": pred_probs,
        "classes":    label_encoder.classes_,
        "img_bgr":    img_bgr,
        "mask":       mask_final,
        "jenis_latar": jenis_latar,
    })
    return hasil


def buat_visualisasi(hasil_prediksi):
    """
    Draw bounding box + label on tightly cropped coffee bean image ROI.
    Returns img_crop (numpy array BGR).
    """
    img_bgr = hasil_prediksi['img_bgr']
    h_img, w_img = img_bgr.shape[:2]
    pred_label = hasil_prediksi['pred_label']
    pred_probs = hasil_prediksi['pred_probs']
    mask       = hasil_prediksi['mask']

    # Determine bounding box from mask
    if mask is not None and np.any(mask > 0):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_pts = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_pts)
            scale_x = w_img / config.IMG_SIZE
            scale_y = h_img / config.IMG_SIZE
            x, y = int(x * scale_x), int(y * scale_y)
            w, h = int(w * scale_x), int(h * scale_y)
            # Add safe margin (5% of dimensions)
            margin = int(min(w_img, h_img) * 0.05)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(w_img - x, w + 2 * margin)
            h = min(h_img - y, h + 2 * margin)
        else:
            margin_w, margin_h = int(w_img * 0.35), int(h_img * 0.35)
            x, y, w, h = margin_w, margin_h, max(50, w_img - 2*margin_w), max(50, h_img - 2*margin_h)
    else:
        margin_w, margin_h = int(w_img * 0.35), int(h_img * 0.35)
        x, y, w, h = margin_w, margin_h, max(50, w_img - 2*margin_w), max(50, h_img - 2*margin_h)

    # Crop tightly around coffee bean area
    img_crop = img_bgr[y:y+h, x:x+w].copy()
    h_crop, w_crop = img_crop.shape[:2]

    # Color based on coffee variety
    warna = WARNA_PER_KELAS.get(pred_label.lower(), WARNA_DEFAULT)
    max_prob = float(max(pred_probs))

    # Draw bounding box outline
    cv2.rectangle(img_crop, (0, 0), (w_crop - 1, h_crop - 1), warna, 4)

    # Format English variety label for overlay text
    display_label_map = {'arabika': 'ARABICA', 'liberika': 'LIBERICA', 'robusta': 'ROBUSTA'}
    display_label = display_label_map.get(pred_label.lower(), pred_label.upper())

    # Text label
    label_text  = f"{display_label} ({max_prob * 100:.1f}%)"
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = max(0.5, w_crop / 350.0)
    thickness   = max(1, int(w_crop / 300.0))
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Label background banner
    cv2.rectangle(img_crop, (0, 0), (min(w_crop, tw + 14), th + 14), warna, -1)
    cv2.putText(img_crop, label_text,
                (6, th + 8),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_crop


def buat_grid_output(list_img_vis, list_nama, cols=2):
    """
    Combine all visualization images into a single grid PNG.
    """
    if not list_img_vis:
        return None

    max_h = max(img.shape[0] for img in list_img_vis)
    max_w = max(img.shape[1] for img in list_img_vis)

    gambar_seragam = []
    for img in list_img_vis:
        resized = cv2.resize(img, (max_w, max_h), interpolation=cv2.INTER_AREA)
        gambar_seragam.append(resized)

    rows = (len(gambar_seragam) + cols - 1) // cols

    while len(gambar_seragam) < rows * cols:
        gambar_seragam.append(np.full((max_h, max_w, 3), 40, dtype=np.uint8))

    baris_list = []
    for r in range(rows):
        baris = np.hstack(gambar_seragam[r * cols: (r + 1) * cols])
        baris_list.append(baris)

    grid = np.vstack(baris_list)
    return grid


# ============================================================
# MAIN
# ============================================================
print("\n" + "=" * 60)
print("  MULTI-IMAGE COFFEE BEAN PREDICTION CHECK")
print("=" * 60)

# Load model
scaler, model, label_encoder = muat_model()
print(f"  [OK] Model loaded. Classes: {list(label_encoder.classes_)}\n")

os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
hasil_semua   = []
img_vis_semua = []
nama_semua    = []

for i, path_gambar in enumerate(DAFTAR_GAMBAR):
    nama_file = os.path.basename(path_gambar)
    print(f"[{i+1}/{len(DAFTAR_GAMBAR)}] Processing: {nama_file}")

    # Predict
    hasil = prediksi_gambar(path_gambar, scaler, model, label_encoder)

    if not hasil['berhasil']:
        print(f"      --> SKIPPED (failed)\n")
        continue

    pred_label = hasil['pred_label']
    pred_probs = hasil['pred_probs']
    max_prob   = float(max(pred_probs))

    display_label_map = {'arabika': 'Arabica', 'liberika': 'Liberica', 'robusta': 'Robusta'}
    display_pred = display_label_map.get(pred_label.lower(), pred_label.capitalize())

    # Print results to terminal
    print(f"  Image Size  : {hasil['img_bgr'].shape[1]}x{hasil['img_bgr'].shape[0]} px")
    print(f"  Prediction  : {display_pred.upper()}")
    print(f"  Probabilities:")
    for cls, prob in zip(label_encoder.classes_, pred_probs):
        cls_name = display_label_map.get(cls.lower(), cls.capitalize())
        bar    = "=" * int(prob * 25)
        marker = " <-- PREDICTION" if cls == pred_label else ""
        print(f"    {cls_name:<12}: {prob*100:6.2f}%  [{bar}]{marker}")
    print()

    # Draw visualization with bounding box
    img_vis = buat_visualisasi(hasil)

    # Save individual image ROI
    nama_output = f"hasil_{os.path.splitext(nama_file)[0]}_roi.png"
    output_path = os.path.join(config.OUTPUTS_DIR, nama_output)
    cv2.imwrite(output_path, img_vis)
    print(f"  [OK] Saved to: {output_path}\n")

    img_vis_semua.append(img_vis)
    nama_semua.append(nama_file)
    hasil_semua.append({
        "file":       nama_file,
        "prediksi":   display_pred,
        "akurasi":    max_prob,
        "proba":      {cls: float(p) for cls, p in zip(label_encoder.classes_, pred_probs)},
    })

# ============================================================
# Create combined grid PNG
# ============================================================
if img_vis_semua:
    grid_img = buat_grid_output(img_vis_semua, nama_semua, cols=2)
    grid_path = os.path.join(config.OUTPUTS_DIR, "hasil_cek_semua_gambar.png")
    cv2.imwrite(grid_path, grid_img)

    print("=" * 60)
    print("  SUMMARY OF ALL PREDICTIONS")
    print("=" * 60)
    for r in hasil_semua:
        status = "✓" if r['akurasi'] >= 0.7 else "?"
        print(f"  {status} {r['file']:<30} --> {r['prediksi'].upper():<10} ({r['akurasi']*100:.1f}%)")
    print()
    print(f"  [OK] Combined grid image saved: {grid_path}")
    print("=" * 60)
else:
    print("[ERROR] No images processed successfully.")

