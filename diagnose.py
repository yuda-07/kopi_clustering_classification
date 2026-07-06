import sys
sys.path.insert(0, 'd:/AI/kopi_clustering_classification')

import os
import cv2
import pandas as pd
import numpy as np
import config

print("=== Diagnostic Script: Checking Image Loading & Masking ===\n")

df = pd.read_csv(config.LABELED_DATASET_CSV)
images_dir = config.COFFEE_IMAGES_DIR

def cari_path(nama_file):
    for subfolder in ['arabika', 'robusta', 'liberika']:
        path = os.path.join(images_dir, subfolder, nama_file)
        if os.path.exists(path):
            return path
    return None

for cid in range(8):
    baris = df[df['cluster_label'] == cid]
    print(f"Cluster {cid} ({config.get_cluster_name(cid)}): total {len(baris)} images")
    
    # Ambil 3 sampel
    sampel = baris.head(3)
    for idx, row in sampel.iterrows():
        nama_file = row['nama_file']
        path = cari_path(nama_file)
        if path is None:
            print(f"  [ERROR] {nama_file} tidak ditemukan di folder!")
            continue
            
        img = cv2.imread(path)
        if img is None:
            print(f"  [ERROR] Gagal membaca file {nama_file}!")
            continue
            
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        brightness = pixels.mean(axis=1)
        
        mask_200 = brightness < 200
        mask_220 = brightness < 220
        mask_230 = brightness < 230
        
        print(f"  File: {nama_file} (shape={h}x{w})")
        print(f"    Piksel < 200: {mask_200.sum()} (mean RGB={pixels[mask_200].mean(axis=0).astype(int) if mask_200.sum() > 0 else 'N/A'})")
        print(f"    Piksel < 220: {mask_220.sum()} (mean RGB={pixels[mask_220].mean(axis=0).astype(int) if mask_220.sum() > 0 else 'N/A'})")
        print(f"    Piksel < 230: {mask_230.sum()} (mean RGB={pixels[mask_230].mean(axis=0).astype(int) if mask_230.sum() > 0 else 'N/A'})")

print("\nSELESAI!")
