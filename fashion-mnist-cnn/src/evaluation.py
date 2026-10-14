"""Fase 5 - Evaluación del modelo."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .config import CLASS_NAMES_ES, NUM_CLASSES


def evaluate_test(model, prepared: dict) -> tuple[float, float]:
    loss, acc = model.evaluate(prepared["x_test"], prepared["y_test"], verbose=0)
    return float(loss), float(acc)


def predict_proba(model, x) -> np.ndarray:
    return model.predict(x, verbose=0)


def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    epochs = range(1, len(history.history["loss"]) + 1)
    axes[0].plot(epochs, history.history["loss"], "o-", label="Entrenamiento")
    axes[0].plot(epochs, history.history["val_loss"], "s-", label="Validación")
    axes[0].set_title("Pérdida (loss)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history.history["accuracy"], "o-", label="Entrenamiento")
    axes[1].plot(epochs, history.history["val_accuracy"], "s-", label="Validación")
    axes[1].set_title("Exactitud (accuracy)")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    return fig


def compute_confusion(model, prepared: dict):
    proba = predict_proba(model, prepared["x_test"])
    y_pred = proba.argmax(axis=1)
    cm = confusion_matrix(prepared["y_test"], y_pred, labels=range(NUM_CLASSES))
    report = classification_report(
        prepared["y_test"],
        y_pred,
        labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES_ES,
        zero_division=0,
    )
    return y_pred, cm, report


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES_ES,
        yticklabels=CLASS_NAMES_ES,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión - conjunto de prueba")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    return fig


def top_confusions(cm, top: int = 5) -> list[dict]:
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    return [
        {"real": CLASS_NAMES_ES[i], "predicho": CLASS_NAMES_ES[j], "casos": count}
        for count, i, j in pairs[:top]
    ]


def per_class_accuracy(cm) -> list[dict]:
    rows = []
    for i in range(NUM_CLASSES):
        total = cm[i].sum()
        correct = cm[i, i]
        rows.append(
            {
                "Clase": CLASS_NAMES_ES[i],
                "Correctas": int(correct),
                "Total": int(total),
                "Accuracy": round(100 * correct / total, 2) if total else 0.0,
            }
        )
    return rows


def plot_predictions_grid(model, x, y, n: int = 16, only_errors: bool = False):
    proba = predict_proba(model, x)
    y_pred = proba.argmax(axis=1)
    if only_errors:
        pool = np.where(y_pred != y)[0]
        title = "Errores de clasificación"
    else:
        pool = np.arange(len(x))
        title = "Predicciones sobre el conjunto de prueba"
    idx = pool[:n]
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        ax.imshow(x[i, ..., 0], cmap="gray")
        color = "green" if y_pred[i] == y[i] else "red"
        ax.set_title(
            f"Real: {CLASS_NAMES_ES[y[i]]}\nPred: {CLASS_NAMES_ES[y_pred[i]]} ({proba[i].max():.0%})",
            fontsize=8,
            color=color,
        )
        ax.axis("off")
    for ax in axes[len(idx):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
