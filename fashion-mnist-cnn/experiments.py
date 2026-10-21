"""Fase C del taller: experimentación comparativa de la CNN.

Entrena varias configuraciones para comparar el impacto de:
  - Dropout (con / sin)
  - Optimizador (Adam vs SGD)
  - Data Augmentation (con / sin)
  - Batch Normalization (innovación)

Guarda en `results/`:
  - experiments.json  (historiales + métricas + config)
  - fig_curvas_<slug>.png (Loss/Accuracy train vs val)
  - fig_confusion_final.png (matriz de confusión del modelo final)
  - modelo_final.keras

Uso:
    .venv\\Scripts\\python.exe experiments.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import DEFAULT_HPARAMS, RANDOM_SEED
from src.data_preparation import get_prepared_data
from src.evaluation import (
    compute_confusion,
    evaluate_test,
    per_class_accuracy,
    plot_confusion_matrix,
    top_confusions,
)
from src.modeling import build_cnn, save_model

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 20

EXPERIMENTS = [
    {"slug": "sin_dropout", "titulo": "Sin Dropout", "use_dropout": False, "use_augmentation": False, "use_batchnorm": False, "optimizer": "adam"},
    {"slug": "con_dropout", "titulo": "Con Dropout", "use_dropout": True, "use_augmentation": False, "use_batchnorm": False, "optimizer": "adam"},
    {"slug": "dropout_aug", "titulo": "Dropout + Augmentation", "use_dropout": True, "use_augmentation": True, "use_batchnorm": False, "optimizer": "adam"},
    {"slug": "dropout_aug_sgd", "titulo": "Dropout + Aug + SGD", "use_dropout": True, "use_augmentation": True, "use_batchnorm": False, "optimizer": "sgd"},
    {"slug": "final_bn", "titulo": "Mejorado (+ BatchNorm)", "use_dropout": True, "use_augmentation": True, "use_batchnorm": True, "optimizer": "adam"},
]


def set_seed(seed: int = RANDOM_SEED) -> None:
    import random

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_curves(history, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], "o-", label="Entrenamiento")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Validación")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["accuracy"], "o-", label="Entrenamiento")
    axes[1].plot(epochs, history["val_accuracy"], "s-", label="Validación")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Exactitud")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    set_seed()
    print("Preparando datos...")
    prepared = get_prepared_data(val_size=DEFAULT_HPARAMS["validation_split"])
    hparams = {**DEFAULT_HPARAMS, "epochs": EPOCHS}

    results = {"epochs": EPOCHS, "hparams": hparams, "experiments": []}
    best_acc = -1.0
    best_model = None
    best_name = ""

    for exp in EXPERIMENTS:
        print(f"\n>>> Entrenando: {exp['titulo']} ({exp['optimizer']})")
        model = build_cnn(
            hparams,
            use_augmentation=exp["use_augmentation"],
            use_dropout=exp["use_dropout"],
            use_batchnorm=exp["use_batchnorm"],
        )
        n_params = int(model.count_params())
        start = time.perf_counter()
        history = train_model_wrapper(model, prepared, hparams, exp["optimizer"])
        elapsed = time.perf_counter() - start
        loss, acc = evaluate_test(model, prepared)
        results["experiments"].append(
            {
                "slug": exp["slug"],
                "titulo": exp["titulo"],
                "optimizer": exp["optimizer"],
                "use_dropout": exp["use_dropout"],
                "use_augmentation": exp["use_augmentation"],
                "use_batchnorm": exp["use_batchnorm"],
                "params": n_params,
                "segundos": round(elapsed, 1),
                "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
                "test_loss": round(loss, 4),
                "test_accuracy": round(acc, 4),
                "val_accuracy_final": round(float(history.history["val_accuracy"][-1]), 4),
                "train_accuracy_final": round(float(history.history["accuracy"][-1]), 4),
                "val_loss_final": round(float(history.history["val_loss"][-1]), 4),
            }
        )
        plot_curves(history.history, exp["titulo"], RESULTS_DIR / f"fig_curvas_{exp['slug']}.png")
        print(f"    test_acc={acc:.4f}  val_acc(final)={history.history['val_accuracy'][-1]:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = exp["titulo"]

    results["best"] = {"titulo": best_name, "test_accuracy": round(best_acc, 4)}
    save_model(best_model, RESULTS_DIR / "modelo_final.keras")

    # Matriz de confusión + reporte del MEJOR modelo
    print(f"\nMejor modelo: {best_name} (acc={best_acc:.4f}). Calculando matriz de confusión...")
    y_pred, cm, report = compute_confusion(best_model, prepared)
    np.save(RESULTS_DIR / "confusion_matrix.npy", cm)
    fig = plot_confusion_matrix(cm)
    fig.savefig(RESULTS_DIR / "fig_confusion_final.png", dpi=130)
    plt.close(fig)

    correct = int(np.trace(cm))
    total = int(cm.sum())
    results["accuracy_formula"] = {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4),
    }
    results["top_confusions"] = top_confusions(cm)
    results["per_class"] = per_class_accuracy(cm)
    results["classification_report"] = report
    results["confusion_matrix"] = cm.tolist()

    (RESULTS_DIR / "experiments.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nListo. Resultados en:", RESULTS_DIR)


def train_model_wrapper(model, prepared, hparams, optimizer):
    from src.modeling import train_model

    return train_model(model, prepared, hparams, optimizer=optimizer, early_stopping=True)


if __name__ == "__main__":
    main()
