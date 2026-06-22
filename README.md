kopi_clustering_classification/
│
├── data/
│   ├── raw/
│   │   └── data_kopi_raw.csv
│   ├── processed/
│   │   └── data_kopi_processed.csv
│   └── hasil/
│       ├── hasil_kmeans.csv
│       └── hasil_naive_bayes.csv
│
├── notebooks/
│   ├── 01_eksplorasi_data.ipynb
│   ├── 02_preprocessing_data.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   ├── 04_naive_bayes_klasifikasi.ipynb
│   └── 05_evaluasi_dan_visualisasi.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── kmeans_model.py
│   ├── naive_bayes_model.py
│   ├── evaluasi.py
│   └── visualisasi.py
│
├── models/
│   ├── kmeans_model.pkl
│   └── naive_bayes_model.pkl
│
├── outputs/
│   ├── grafik/
│   │   ├── elbow_method.png
│   │   ├── cluster_plot.png
│   │   └── confusion_matrix.png
│   └── laporan/
│       └── laporan_hasil.pdf
│
├── requirements.txt
├── README.md
└── main.py