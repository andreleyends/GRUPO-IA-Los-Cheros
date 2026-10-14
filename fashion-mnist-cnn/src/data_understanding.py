"""Fase 2 - Comprensión de los Datos (EDA)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES_ES, NUM_CLASSES


def dataset_summary(x_train, y_train, x_test, y_test) -> dict:
    return {
        "train_images": x_train.shape[0],
        "test_images": x_test.shape[0],
        "image_shape": f"{x_train.shape[1]}x{x_train.shape[2]} (escala de grises)",
        "pixel_dtype": str(x_train.dtype),
        "pixel_range": f"{int(x_train.min())} - {int(x_train.max())}",
        "num_classes": len(np.unique(y_train)),
        "train_balance": np.bincount(y_train, minlength=NUM_CLASSES),
        "test_balance": np.bincount(y_test, minlength=NUM_CLASSES),
        "pixel_mean": float(x_train.mean()),
        "pixel_std": float(x_train.std()),
        "missing_values": int(np.isnan(x_train).sum()),
    }


def class_balance_table(y_train, y_test) -> list[dict]:
    train_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    test_counts = np.bincount(y_test, minlength=NUM_CLASSES)
    rows = []
    for i, name in enumerate(CLASS_NAMES_ES):
        rows.append(
            {
                "Clase": i,
                "Etiqueta": name,
                "Train": int(train_counts[i]),
                "Test": int(test_counts[i]),
                "% Train": round(100 * train_counts[i] / train_counts.sum(), 2),
            }
        )
    return rows


def plot_class_distribution(y_train, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    train_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    test_counts = np.bincount(y_test, minlength=NUM_CLASSES)
    positions = np.arange(NUM_CLASSES)
    colors = plt.cm.tab10(positions / NUM_CLASSES)

    axes[0].bar(positions, train_counts, color=colors)
    axes[0].set_title("Distribución de clases - Entrenamiento")
    axes[1].bar(positions, test_counts, color=colors)
    axes[1].set_title("Distribución de clases - Prueba")
    for ax, counts in zip(axes, (train_counts, test_counts)):
        ax.set_xticks(positions)
        ax.set_xticklabels(CLASS_NAMES_ES, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Cantidad de imágenes")
        ax.axhline(counts.mean(), color="red", linestyle="--", linewidth=1, label=f"Media={counts.mean():.0f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_samples_grid(x, y, rows: int = 5, cols: int = 8):
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 1.6))
    indices = np.random.RandomState(42).choice(len(x), rows * cols, replace=False)
    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(x[idx], cmap="gray")
        ax.set_title(f"{y[idx]} - {CLASS_NAMES_ES[y[idx]]}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_one_per_class(x, y):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(16, 2.2))
    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0][0]
        axes[cls].imshow(x[idx], cmap="gray")
        axes[cls].set_title(f"{cls}\n{CLASS_NAMES_ES[cls]}", fontsize=7)
        axes[cls].axis("off")
    fig.tight_layout()
    return fig


def plot_intensity_histogram(x):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(x.ravel(), bins=50, color="#4C72B0")
    ax.set_title("Distribución de intensidad de píxeles (entrenamiento)")
    ax.set_xlabel("Valor de píxel (0-255)")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def plot_mean_image_per_class(x, y):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for cls in range(NUM_CLASSES):
        mean_img = x[y == cls].mean(axis=0)
        ax = axes.ravel()[cls]
        ax.imshow(mean_img, cmap="viridis")
        ax.set_title(CLASS_NAMES_ES[cls], fontsize=8)
        ax.axis("off")
    fig.suptitle("Imagen promedio por clase", fontsize=11)
    fig.tight_layout()
    return fig
