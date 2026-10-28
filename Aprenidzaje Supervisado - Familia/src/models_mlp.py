"""
Modelos MLPRegressor para regresión.

Incluye búsqueda de hiperparámetros, análisis de curvas de aprendizaje,
pérdida por iteración y permutation importance.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def crear_pipeline_mlp(
    hidden_layer_sizes: tuple = (100,),
    activation: str = "relu",
    alpha: float = 0.0001,
    learning_rate_init: float = 0.001,
    solver: str = "adam",
    early_stopping: bool = True,
    max_iter: int = 500,
    random_state: int = 42,
) -> Pipeline:
    """
    Crea pipeline para MLPRegressor con escalado (obligatorio para redes neuronales).
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        solver=solver,
        early_stopping=early_stopping,
        max_iter=max_iter,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", mlp),
    ])


def obtener_espacio_hiperparametros_mlp() -> Dict[str, list]:
    """Espacio de búsqueda para GridSearchCV."""
    return {
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.01],
        "mlp__solver": ["adam", "sgd"],
        "mlp__early_stopping": [True],
    }
