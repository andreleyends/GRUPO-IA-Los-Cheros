"""
Métricas de evaluación, reportes y utilidades para gráficas.

Guarda resultados en CSV/PNG con alta resolución.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from . import config


def evaluar_clasificacion(
    y_true,
    y_pred,
    y_proba=None,
    nombre_modelo: str = "modelo",
) -> Dict[str, Any]:
    """Evalúa un modelo de clasificación y devuelve un diccionario de métricas."""
    resultados = {
        "modelo": nombre_modelo,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        try:
            clases = np.unique(y_true)
            if len(clases) == 2:
                proba = y_proba[:, 1] if np.ndim(y_proba) == 2 else y_proba
                resultados["roc_auc"] = float(roc_auc_score(y_true, proba))
            else:
                resultados["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
                resultados["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception as exc:
            print(f"Advertencia ROC-AUC: {exc}")

    return resultados


def evaluar_regresion(y_true, y_pred, nombre_modelo: str = "modelo") -> Dict[str, Any]:
    """Evalúa un modelo de regresión y devuelve un diccionario de métricas."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "modelo": nombre_modelo,
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def guardar_matriz_confusion(y_true, y_pred, nombre: str, labels=None):
    """Guarda la matriz de confusión como PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels = list(range(cm.shape[0]))
    plt.figure(figsize=config.FIGSIZE_STD)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.ylabel("Verdadero")
    plt.xlabel("Predicho")
    plt.tight_layout()
    ruta = config.FIGURES_DIR / f"confusion_{nombre.replace(' ', '_')}.png"
    plt.savefig(ruta, dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_curvas_roc(y_true, y_proba, nombre: str, clases=None):
    """Guarda curvas ROC para clasificación binaria o multiclase (One-vs-Rest)."""
    plt.figure(figsize=config.FIGSIZE_STD)
    if clases is None:
        clases = np.unique(y_true)

    if len(clases) == 2:
        proba = y_proba[:, 1] if np.ndim(y_proba) == 2 else y_proba
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=clases)
        for i, clase in enumerate(clases):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, label=f"Clase {clase} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title(f"Curvas ROC - {nombre}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = config.FIGURES_DIR / f"roc_{nombre.replace(' ', '_')}.png"
    plt.savefig(ruta, dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_real_vs_predicho(y_true, y_pred, nombre: str):
    """Gráfico real vs predicho para regresión."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    plt.figure(figsize=config.FIGSIZE_STD)
    plt.scatter(y_true, y_pred, alpha=0.5)
    minimo = min(y_true.min(), y_pred.min())
    maximo = max(y_true.max(), y_pred.max())
    plt.plot([minimo, maximo], [minimo, maximo], "r--", lw=2, label="Ideal")
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.title(f"Real vs Predicho - {nombre}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = config.FIGURES_DIR / f"real_vs_pred_{nombre.replace(' ', '_')}.png"
    plt.savefig(ruta, dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_residuos(y_true, y_pred, nombre: str):
    """Gráfico de residuos para regresión."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuos = y_true - y_pred
    plt.figure(figsize=config.FIGSIZE_STD)
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="--", lw=2)
    plt.xlabel("Valores Predichos")
    plt.ylabel("Residuos")
    plt.title(f"Gráfico de Residuos - {nombre}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = config.FIGURES_DIR / f"residuos_{nombre.replace(' ', '_')}.png"
    plt.savefig(ruta, dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_curva_perdida(mlp_pipeline, nombre: str):
    """Guarda la curva de pérdida por iteración de un MLP."""
    mlp = mlp_pipeline.named_steps.get("mlp")
    if mlp is not None and hasattr(mlp, "loss_curve_"):
        plt.figure(figsize=config.FIGSIZE_STD)
        plt.plot(mlp.loss_curve_)
        plt.xlabel("Iteración")
        plt.ylabel("Pérdida")
        plt.title(f"Curva de Pérdida - {nombre}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        ruta = config.FIGURES_DIR / f"loss_curve_{nombre.replace(' ', '_')}.png"
        plt.savefig(ruta, dpi=config.DPI_FIG, bbox_inches="tight")
        plt.close()
        return ruta
    return None


def guardar_fronteras_decision_2d(
    modelos: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    nombre: str,
    titulos: Dict[str, str] | None = None,
) -> Path:
    """Dibuja las fronteras de decisión 2D de varios modelos lado a lado."""
    from matplotlib.colors import ListedColormap

    n_modelos = len(modelos)
    fig, axes = plt.subplots(1, n_modelos, figsize=(6 * n_modelos, 5))
    if n_modelos == 1:
        axes = [axes]

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    cmap_puntos = ListedColormap(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])

    for ax, (nombre_modelo, modelo) in zip(axes, modelos.items()):
        try:
            Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error: {exc}", ha="center", va="center")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_puntos, edgecolor="k", s=25)
        ax.set_title((titulos or {}).get(nombre_modelo, nombre_modelo))
        ax.set_xlabel("Componente / característica 1")
        ax.set_ylabel("Componente / característica 2")

    plt.tight_layout()
    ruta = config.FIGURES_DIR / f"fronteras_{nombre.replace(' ', '_')}.png"
    plt.savefig(ruta, dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_curva_alpha_nb(
    resultados: pd.DataFrame,
    nombre: str = "sensibilidad_alpha_nb",
) -> Path:
    """Gráfica del efecto de alpha en NB (F1 macro de validación cruzada)."""
    plt.figure(figsize=config.FIGSIZE_STD)
    for modelo, grupo in resultados.groupby("modelo"):
        grupo = grupo.sort_values("alpha")
        plt.plot(grupo["alpha"], grupo["f1_macro"], marker="o", label=modelo)
    plt.xscale("log")
    plt.xlabel("Alpha (suavizado)")
    plt.ylabel("F1 macro (validación cruzada)")
    plt.title("Sensibilidad de Naive Bayes al suavizado alpha")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ruta = config.FIGURES_DIR / f"{nombre}.png"
    plt.savefig(ruta, dpi=config.DPI_FIG, bbox_inches="tight")
    plt.close()
    return ruta


def guardar_classification_report(
    y_true,
    y_pred,
    nombre: str,
    etiquetas: list[str] | None = None,
) -> Path:
    """Guarda el classification_report como CSV y lo imprime en consola."""
    reporte = classification_report(
        y_true, y_pred, target_names=etiquetas, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(reporte).transpose()
    ruta = config.TABLES_DIR / f"classification_report_{nombre.replace(' ', '_')}.csv"
    df.to_csv(ruta, encoding="utf-8-sig")
    print(classification_report(y_true, y_pred, target_names=etiquetas, zero_division=0))
    return ruta
