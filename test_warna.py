"""
Script test: ambil gambar representatif per cluster dan simpan untuk verifikasi cropping.
"""
import sys
sys.path.insert(0, 'd:/AI/kopi_clustering_classification')

import os
import cv2
import pandas as pd
import config
from src.visualization import _ambil_gambar_representatif

print("=== Test: Cropping Biji Kopi per Cluster ===\n")

df_labeled = pd.read_csv(config.LABELED_DATASET_CSV)
images_dir = config.COFFEE_IMAGES_DIR

os.makedirs("outputs/test_crops", exist_ok=True)

for cid in range(8):
    nama = config.get_cluster_name(cid)
    img = _ambil_gambar_representatif(
        cluster_id=cid,
        df_labeled=df_labeled,
        images_dir=images_dir,
        ukuran=(120, 120)
    )
    if img is not None:
        # Convert RGB back to BGR for cv2.imwrite
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_path = f"outputs/test_crops/cluster_{cid}_{nama.replace(' ', '_')}.png"
        cv2.imwrite(save_path, img_bgr)
        print(f"  Cluster {cid} ({nama:15s}): BERHASIL disimpan ke {save_path}")
    else:
        print(f"  Cluster {cid} ({nama:15s}): GAGAL")

print("\nSELESAI! Silakan periksa folder outputs/test_crops/ untuk melihat hasil cropping.")
