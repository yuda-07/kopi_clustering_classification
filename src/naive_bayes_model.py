"""
naive_bayes_model.py - Modul Naive Bayes Classification
=========================================================
Melakukan klasifikasi kualitas biji kopi menggunakan Gaussian Naive Bayes:
- Split data menjadi train (80%) dan test (20%)
- Training model GaussianNB
- Prediksi pada data test
- Evaluasi: Akurasi, Precision, Recall, F1
- Simpan model ke file .pkl
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def siapkan_data_training(df_fitur, kolom_label='cluster_label'):
    """
    Menyiapkan data fitur (X) dan label (y) untuk training Naive Bayes.

    Parameter:
        df_fitur (pd.DataFrame): DataFrame berisi fitur dan label.
        kolom_label (str): Nama kolom yang berisi label kelas.

    Return:
        tuple: (X: np.ndarray, y: np.ndarray, nama_fitur: list, label_encoder: LabelEncoder)
    """
    try:
        # Pisahkan kolom non-fitur dan label
        kolom_buang = ['nama_file', 'cluster_label', 'jenis_kopi']
        nama_fitur = [c for c in df_fitur.columns if c not in kolom_buang]

        X = df_fitur[nama_fitur].values
        y = df_fitur[kolom_label].values

        # Encode label jika berupa string (GaussianNB butuh numerik)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        print(f"[OK] Training data prepared: X={X.shape}, y={y_encoded.shape}")
        print(f"     Unique classes: {label_encoder.classes_}")
        print(f"     Number of features: {len(nama_fitur)}")

        return X, y_encoded, nama_fitur, label_encoder

    except Exception as e:
        print(f"[ERROR] Failed to prepare training data: {e}")
        return None, None, None, None


def training_naive_bayes(X, y, test_size=None, random_state=None):
    """
    Trains Gaussian Naive Bayes model.
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    print(f"[INFO] Data split: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        print("[INFO] Stratified split failed, using regular split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    print(f"     Train samples: {X_train.shape[0]}")
    print(f"     Test samples : {X_test.shape[0]}")

    model = GaussianNB()
    model.fit(X_train, y_train)
    print("[OK] Gaussian Naive Bayes training complete.")

    y_pred = model.predict(X_test)

    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
    }


def training_random_forest(X, y, scaler=None, feature_names=None, save_scaler_path=None, test_size=None, random_state=None, n_estimators=100):
    """
    Trains Random Forest model.
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    print(f"[INFO] Data split: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        print("[INFO] Stratified split failed, using regular split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    print(f"     Train samples: {X_train.shape[0]}")
    print(f"     Test samples : {X_test.shape[0]}")

    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    if feature_names is not None and not hasattr(scaler, 'feature_names_'):
        try:
            scaler.feature_names_ = list(feature_names)
        except Exception:
            pass

    model = RandomForestClassifier(
        n_estimators=n_estimators if n_estimators != 200 else 300,
        class_weight='balanced',
        random_state=random_state
    )
    model.fit(X_train_scaled, y_train)
    print("[OK] Random Forest training complete.")

    if save_scaler_path is not None:
        try:
            joblib.dump(scaler, save_scaler_path)
            print(f"[OK] Random Forest scaler saved to: {save_scaler_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save Random Forest scaler: {e}")

    y_pred = model.predict(X_test_scaled)

    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
    }


def evaluasi_model(y_test, y_pred, label_encoder=None):
    """
    Evaluates classification model performance.
    """
    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    hasil = {
        'akurasi': akurasi,
        'presisi': presisi,
        'recall': recall,
        'f1_score': f1,
    }

    print("=" * 40)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Accuracy  : {akurasi:.4f} ({akurasi*100:.2f}%)")
    print(f"  Precision : {presisi:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("=" * 40)

    return hasil


def prediksi_baru(model, scaler, fitur_baru):
    """
    Makes predictions on new data.
    """
    try:
        fitur_scaled = scaler.transform(fitur_baru.reshape(1, -1))
        prediksi = model.predict(fitur_scaled)
        return prediksi[0]
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return None


def simpan_model_nb(model, path_output=None):
    """
    Saves model to a .pkl file.
    """
    if path_output is None:
        path_output = config.NAIVE_BAYES_MODEL_PATH
    try:
        joblib.dump(model, path_output)
        print(f"[OK] Classifier model saved to: {path_output}")
    except Exception as e:
        print(f"[ERROR] Failed to save classifier model: {e}")


def muat_model_nb(path_model=None):
    """
    Loads model from a .pkl file.
    """
    if path_model is None:
        path_model = config.NAIVE_BAYES_MODEL_PATH
    try:
        model = joblib.load(path_model)
        print(f"[OK] Classifier model loaded from: {path_model}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load classifier model: {e}")
        return None


# ============================================================
# TEST MODUL (jalankan langsung)
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: naive_bayes_model.py")
    print("=" * 50)

    # Buat data dummy
    np.random.seed(config.RANDOM_STATE)
    n_sampel = 200
    n_fitur = 15
    X_dummy = np.random.randn(n_sampel, n_fitur)
    y_dummy = np.random.choice([0, 1, 2], size=n_sampel)

    # Training
    hasil = training_naive_bayes(X_dummy, y_dummy)

    # Evaluasi
    if hasil:
        eval_result = evaluasi_model(hasil['y_test'], hasil['y_pred'])

    print("=" * 50)
    print("Test naive_bayes_model selesai.")
