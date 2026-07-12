"""
predict.py - Skrip Prediksi Jenis Biji Kopi
===========================================
Mengidentifikasi jenis biji kopi (Arabika, Robusta, Liberica) dari file gambar input.
Penggunaan:
    python predict.py <path_ke_gambar|path_ke_folder|all> [output_excel_path]

Contoh:
    python predict.py all
    python predict.py dataset/raw/coffee_images/arabika/arabika_1.jpg
    python predict.py dataset/raw/coffee_images predictions.xlsx
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import joblib

# Pastikan path proyek terdaftar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src import image_preprocessing
from src import feature_extraction


def get_image_paths(path_input):
    """Kembalikan daftar path gambar untuk file tunggal, folder, atau keyword all."""
    if path_input.lower() == 'all':
        path_input = config.COFFEE_IMAGES_DIR

    if os.path.isfile(path_input):
        return [path_input]

    if os.path.isdir(path_input):
        image_paths = []
        for root, _, files in os.walk(path_input):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in config.IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, fname))
        return image_paths

    return []


def build_feature_dataframe(fitur_dict, scaler):
    """Bangun DataFrame fitur sesuai dengan nama fitur yang disimpan di scaler."""
    if hasattr(scaler, 'feature_names_'):
        feature_names = list(scaler.feature_names_)
    elif hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = [k for k in fitur_dict.keys() if k != 'nama_file']

    feature_vector = []
    for name in feature_names:
        if name not in fitur_dict:
            raise ValueError(f"Fitur penting '{name}' tidak ditemukan pada gambar input.")
        feature_vector.append(fitur_dict[name])

    return pd.DataFrame([feature_vector], columns=feature_names)


def normalize_features(scaler, feature_df):
    """Transform fitur menggunakan scaler sambil menjaga atribut feature_names."""
    if hasattr(scaler, 'feature_names_in_'):
        saved_feature_names = scaler.feature_names_in_
        delattr(scaler, 'feature_names_in_')
        try:
            return scaler.transform(feature_df)
        finally:
            scaler.feature_names_in_ = saved_feature_names
    return scaler.transform(feature_df)


def predict_image(path_gambar, scaler, model, label_encoder):
    """Lakukan prediksi jenis kopi untuk satu gambar dan kembalikan hasilnya."""
    img_bgr = cv2.imread(path_gambar)
    if img_bgr is None:
        raise ValueError(f"Gagal membaca gambar: {path_gambar}. File mungkin rusak.")

    hasil_prep = image_preprocessing.preprocessing_gambar(img_bgr)
    if hasil_prep is None:
        raise ValueError("Gagal melakukan preprocessing gambar.")

    fitur_dict = feature_extraction.ekstraksi_fitur_gambar(hasil_prep, nama_file=os.path.basename(path_gambar))
    if fitur_dict is None:
        raise ValueError("Gagal melakukan ekstraksi fitur gambar.")

    feature_df = build_feature_dataframe(fitur_dict, scaler)
    features_scaled = normalize_features(scaler, feature_df)

    pred_class = model.predict(features_scaled)[0]
    pred_probs = model.predict_proba(features_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    classes = label_encoder.classes_
    prob_map = {f"prob_{cls}": float(prob) for cls, prob in zip(classes, pred_probs)}
    prob_map.update({
        'predicted_label': pred_label,
        'source_file': os.path.basename(path_gambar),
        'source_path': os.path.abspath(path_gambar)
    })
    return prob_map


def save_predictions_to_excel(rows, output_path):
    """Simpan hasil prediksi ke file Excel."""
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        df.to_excel(output_path, index=False)
        print(f"[OK] Hasil prediksi disimpan ke: {output_path}")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan Excel: {e}")
        csv_path = os.path.splitext(output_path)[0] + '.csv'
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Sebagai fallback, hasil disimpan ke CSV: {csv_path}")


def main():
    # 1. Validasi input argumen
    if len(sys.argv) < 2:
        print("[ERROR] Path gambar atau keyword 'all' tidak diberikan!")
        print("Penggunaan: python predict.py <path_ke_gambar|path_ke_folder|all> [output_excel_path]")
        print("Contoh: python predict.py all")
        print("Contoh: python predict.py dataset/raw/coffee_images arabika_predictions.xlsx")
        sys.exit(1)

    path_input = sys.argv[1]
    output_path = config.PREDICTIONS_EXCEL_PATH
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    image_paths = get_image_paths(path_input)
    if len(image_paths) == 0:
        print(f"[ERROR] Tidak ada gambar yang ditemukan di: {path_input}")
        sys.exit(1)

    required_files = [
        config.VARIETY_SCALER_PATH,
        config.VARIETY_MODEL_PATH,
        config.VARIETY_LABEL_ENCODER_PATH
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("[ERROR] Model pendukung belum lengkap. Silakan latih model terlebih dahulu dengan menjalankan main.py!")
        print("File yang hilang:")
        for mf in missing_files:
            print(f"  - {os.path.basename(mf)}")
        sys.exit(1)

    scaler = joblib.load(config.VARIETY_SCALER_PATH)
    model = joblib.load(config.VARIETY_MODEL_PATH)
    label_encoder = joblib.load(config.VARIETY_LABEL_ENCODER_PATH)

    results = []
    total = len(image_paths)
    print(f"[INFO] Menjalankan prediksi untuk {total} gambar...")

    for index, image_path in enumerate(image_paths, start=1):
        try:
            print(f"[INFO] ({index}/{total}) Memproses: {image_path}")
            row = predict_image(image_path, scaler, model, label_encoder)
            results.append(row)
        except Exception as e:
            print(f"[ERROR] Gagal memproses {image_path}: {e}")

    if len(results) == 0:
        print("[ERROR] Tidak ada prediksi yang berhasil dibuat.")
        sys.exit(1)

    save_predictions_to_excel(results, output_path)
    print(f"[OK] Prediksi selesai untuk {len(results)} gambar.")


if __name__ == "__main__":
    main()
