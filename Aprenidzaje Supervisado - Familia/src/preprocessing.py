"""
Preprocesamiento con pipelines para evitar fuga de datos.

- División estratificada 80/20 para clasificación, 80/20 para regresión.
- StandardScaler donde aplique (MLP; opcional para LDA/QDA).
- Para Naive Bayes Multinomial NO se escala (requiere conteos/frecuencias no negativas).
- ColumnTransformer para manejar numéricas/categóricas si las hay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from . import config


def dividir_datos(
    X,
    y,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    estratificar: bool = True,
) -> Tuple:
    """Divide datos en entrenamiento y prueba."""
    stratify = y if estratificar and y is not None else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def crear_preprocesador(
    X: pd.DataFrame,
    escalar: bool = True,
) -> ColumnTransformer:
    """
    Crea un preprocesador con ColumnTransformer.
    Escala solo variables numéricas. Para categóricas usa OneHotEncoder.
    """
    if not isinstance(X, pd.DataFrame):
        # Si es numpy array, tratar todo como numérico
        return ColumnTransformer(
            transformers=[("num", StandardScaler() if escalar else "passthrough", slice(None))],
            remainder="drop",
        )
    
    columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    transformers = []
    if columnas_numericas:
        if escalar:
            transformers.append(("num", StandardScaler(), columnas_numericas))
        else:
            transformers.append(("num", "passthrough", columnas_numericas))
    if columnas_categoricas:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), columnas_categoricas))
    
    if not transformers:
        transformers.append(("num", "passthrough", []))
    
    return ColumnTransformer(transformers=transformers, remainder="drop")


def crear_pipeline_modelo(clasificador_o_regresor, preprocesador: Optional[ColumnTransformer] = None):
    """Crea pipeline con preprocesamiento y modelo."""
    if preprocesador is None:
        return clasificador_o_regresor
    return Pipeline([
        ("preprocessor", preprocesador),
        ("model", clasificador_o_regresor),
    ])
