"""Fase 3 - Preparación de los Datos."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """Escala los píxeles de [0, 255] a [0, 1]."""
    return (x.astype("float32") / 255.0)


def add_channel(x: np.ndarray) -> np.ndarray:
    """Añade la dimensión de canal: (n, 28, 28) -> (n, 28, 28, 1)."""
    return x[..., np.newaxis]


def split_train_val(x, y, val_size: float = 0.1, seed: int = 42):
    from sklearn.model_selection import train_test_split

    return train_test_split(x, y, test_size=val_size, random_state=seed, stratify=y)


def build_augmentation():
    """Capa de aumento de datos (Data Augmentation) con Keras."""
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.RandomRotation(0.08, seed=42),
            keras.layers.RandomZoom(0.1, seed=42),
            keras.layers.RandomTranslation(0.1, 0.1, seed=42),
        ],
        name="augmentation",
    )


def build_robust_augmentation():
    """Aumentación reforzada para mayor robustez ante fotos reales.

    Además de la geometría (rotación/zoom/traslación) añade variaciones de
    brillo y contraste que simulan las condiciones de iluminación de fotos
    tomadas con cámara.
    """
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.12),
            keras.layers.RandomTranslation(0.12, 0.12),
            keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
            keras.layers.RandomContrast(0.1),
        ],
        name="robust_augmentation",
    )


def build_production_augmentation():
    """Aumentación de producción: geometría + volteo horizontal + brillo suave.

    Mejoras frente a build_robust_augmentation (que perdía ~2 pts de accuracy
    en test): se elimina el RandomContrast (el preprocesado de inferencia ya
    normaliza el contraste por percentiles) y se añade volteo horizontal,
    seguro para prendas y muy útil para calzado en diagonal.
    """
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomFlip("horizontal", seed=42),
            keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
        ],
        name="production_augmentation",
    )


def get_prepared_data(val_size: float = 0.1):
    """Devuelve datos normalizados, con canal y divididos en train/val/test."""
    from .data_loader import load_fashion_mnist

    x_train, y_train, x_test, y_test = load_fashion_mnist()
    x_train = add_channel(normalize(x_train))
    x_test = add_channel(normalize(x_test))
    x_tr, x_val, y_tr, y_val = split_train_val(x_train, y_train, val_size=val_size)
    return {
        "x_train": x_tr,
        "y_train": y_tr,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def plot_augmentation_preview(x, y, n: int = 8):
    aug = build_augmentation()
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 4))
    indices = np.random.RandomState(0).choice(len(x), n, replace=False)
    batch = x[indices]
    augmented = aug(batch, training=True).numpy()
    for col, idx in enumerate(indices):
        axes[0, col].imshow(batch[col, ..., 0], cmap="gray")
        axes[0, col].set_title(f"Orig\n{y[idx]}", fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(augmented[col, ..., 0], cmap="gray")
        axes[1, col].set_title("Aum.", fontsize=7)
        axes[1, col].axis("off")
    fig.suptitle("Ejemplos originales vs. aumentados", fontsize=11)
    fig.tight_layout()
    return fig


def plot_pipeline_summary(x_train_raw, prepared: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].imshow(x_train_raw[0], cmap="gray")
    axes[0].set_title("Original (0-255)")
    axes[0].axis("off")
    axes[1].imshow(prepared["x_train"][0, ..., 0], cmap="gray")
    axes[1].set_title("Normalizada (0-1)")
    axes[1].axis("off")
    sizes = [len(prepared["x_train"]), len(prepared["x_val"]), len(prepared["x_test"])]
    axes[2].bar(["Train", "Validación", "Prueba"], sizes, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[2].set_title("Tamaño de las particiones")
    for i, v in enumerate(sizes):
        axes[2].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig
