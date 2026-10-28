"""
Modelos LDA y QDA para clasificación.

Incluye análisis discriminante lineal y cuadrático, visualización de fronteras
y proyecciones discriminantes.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def crear_pipeline_lda(
    solver: str = "svd",
    shrinkage: str | float | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Crea pipeline para LDA. Escalado opcional pero recomendado.
    """
    lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda", lda),
    ])


def crear_pipeline_qda(
    reg_param: float = 0.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Crea pipeline para QDA.
    """
    qda = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("qda", qda),
    ])


def obtener_espacio_hiperparametros_lda() -> Dict[str, list]:
    """Espacio de búsqueda para LDA."""
    return {
        "lda__solver": ["svd", "lsqr", "eigen"],
        "lda__shrinkage": [None, "auto", 0.1, 0.5],
    }


def obtener_espacio_hiperparametros_qda() -> Dict[str, list]:
    """Espacio de búsqueda para QDA."""
    return {
        "qda__reg_param": [0.0, 0.1, 0.2, 0.3, 0.5],
    }
