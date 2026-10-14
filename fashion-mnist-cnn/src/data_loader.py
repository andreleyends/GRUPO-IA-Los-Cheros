"""Carga del dataset Fashion-MNIST: CSV local si existe, si no Keras."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IMG_SIZE, TEST_CSV, TRAIN_CSV


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype=np.uint8)
    labels = frame.iloc[:, 0].to_numpy(dtype=np.uint8)
    pixels = frame.iloc[:, 1:].to_numpy(dtype=np.uint8)
    images = pixels.reshape(-1, IMG_SIZE, IMG_SIZE)
    return images, labels


def _load_from_keras() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback para entornos sin los CSV (p. ej. Streamlit Community Cloud)."""
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test


@lru_cache(maxsize=1)
def load_fashion_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (X_train, y_train, X_test, y_test) con imágenes uint8 de 28x28.

    Usa los CSV locales en 'archive/' si existen; en caso contrario descarga el
    dataset con Keras (idéntico a los CSV).
    """
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        x_train, y_train = _read_csv(TRAIN_CSV)
        x_test, y_test = _read_csv(TEST_CSV)
    else:
        x_train, y_train, x_test, y_test = _load_from_keras()
    return x_train, y_train, x_test, y_test


def class_distribution(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=10)
