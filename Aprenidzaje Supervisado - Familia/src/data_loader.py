"""
Carga de datasets para los tres modelos del Equipo 6.

- Naive Bayes (clasificación de texto): 20 Newsgroups (sklearn) con submuestreo
  para crear un conjunto desbalanceado y mostrar la ventaja de ComplementNB.
- MLPRegressor (regresión): California Housing; fallback a diabetes o sintético.
- LDA/QDA (clasificación): Breast Cancer o Wine; además dataset sintético con
  covarianzas distintas por clase para mostrar superioridad de QDA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_california_housing,
    load_breast_cancer,
    load_wine,
    make_classification,
    make_regression,
)
from sklearn.model_selection import train_test_split

from . import config


def _crear_corpus_sintetico_es(
    n_por_clase: int = 120,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[list[str], np.ndarray]:
    """
    Corpus de respaldo en español para el caso en que no haya acceso a
    20 Newsgroups. Genera documentos con vocabulario distintivo por clase.
    """
    rng = np.random.RandomState(random_state)
    vocabularios = {
        "tecnologia": [
            "computadora", "software", "algoritmo", "red", "servidor", "datos",
            "programacion", "sistema", "codigo", "internet", "procesador", "memoria",
        ],
        "deportes": [
            "equipo", "partido", "jugador", "gol", "entrenador", "liga", "torneo",
            "balon", "estadio", "victoria", "campeonato", "defensa",
        ],
        "salud": [
            "medico", "paciente", "tratamiento", "enfermedad", "hospital", "clinica",
            "diagnostico", "medicina", "sintoma", "terapia", "salud", "consulta",
        ],
        "politica": [
            "gobierno", "eleccion", "presidente", "congreso", "ley", "partido",
            "voto", "politica", "ciudadano", "debate", "reforma", "ministro",
        ],
    }
    relleno = ["el", "la", "de", "que", "y", "en", "un", "una", "con", "para", "por", "los"]
    textos: list[str] = []
    etiquetas: list[int] = []
    for idx, (_, palabras) in enumerate(vocabularios.items()):
        for _ in range(n_por_clase):
            n_tokens = rng.randint(30, 80)
            tokens = [palabras[rng.randint(len(palabras))] for _ in range(n_tokens // 2)]
            tokens += [relleno[rng.randint(len(relleno))] for _ in range(n_tokens // 2)]
            rng.shuffle(tokens)
            textos.append(" ".join(tokens))
            etiquetas.append(idx)
    return textos, np.array(etiquetas)


def cargar_20newsgroups(
    categorias: list[str] | None = None,
    subset: str = "all",
    random_state: int = config.RANDOM_STATE,
) -> Tuple[list[str], np.ndarray]:
    """
    Carga 20 Newsgroups para clasificación de texto.

    Si la descarga falla (sin internet), usa un corpus sintético en español
    con cuatro categorías para no interrumpir el experimento.
    """
    if categorias is None:
        categorias = [
            "comp.graphics",
            "rec.sport.baseball",
            "sci.med",
            "talk.politics.mideast",
        ]

    try:
        data = fetch_20newsgroups(
            subset=subset,
            categories=categorias,
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=random_state,
            data_home=str(config.DATA_DIR),
        )
        print(f"20 Newsgroups cargado: {len(data.data)} documentos, {len(data.target_names)} categorías.")
        return data.data, data.target
    except Exception as exc:
        print(f"Advertencia: no se pudo descargar 20 Newsgroups ({exc}).")
        print("Usando corpus sintético en español como respaldo.")
        return _crear_corpus_sintetico_es(random_state=random_state)


def crear_conjunto_desbalanceado_nb(
    textos: list[str],
    y: np.ndarray,
    proporcion_clase_minoritaria: float = 0.1,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[list[str], np.ndarray]:
    """
    Crea un conjunto desbalanceado submuestreando una clase para demostrar
    la ventaja de ComplementNB frente a MultinomialNB.
    """
    rng = np.random.RandomState(random_state)
    clases_unicas = np.unique(y)
    if len(clases_unicas) < 2:
        return textos, y
    
    # Submuestrear la primera clase
    clase_objetivo = clases_unicas[0]
    indices_mayoria = np.where(y != clase_objetivo)[0]
    indices_minoria = np.where(y == clase_objetivo)[0]
    
    n_minoria_deseada = max(1, int(len(indices_mayoria) * proporcion_clase_minoritaria))
    if n_minoria_deseada < len(indices_minoria):
        indices_minoria_sub = rng.choice(indices_minoria, size=n_minoria_deseada, replace=False)
    else:
        indices_minoria_sub = indices_minoria
    
    indices_finales = np.concatenate([indices_mayoria, indices_minoria_sub])
    rng.shuffle(indices_finales)
    
    textos_sub = [textos[i] for i in indices_finales]
    y_sub = y[indices_finales]
    return textos_sub, y_sub


def cargar_california_housing() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga California Housing para regresión con MLPRegressor.
    Si falla la descarga, usa fallback a diabetes o dataset sintético.
    """
    try:
        data = fetch_california_housing(as_frame=True)
        X = data.data
        y = data.target
        return X, y
    except Exception as e:
        print(f"Advertencia: No se pudo descargar California Housing ({e}). Usando fallback.")
        try:
            from sklearn.datasets import load_diabetes
            data = load_diabetes(as_frame=True)
            X = data.data
            y = data.target
            return X, y
        except Exception:
            X, y = make_regression(
                n_samples=20000,
                n_features=8,
                noise=10.0,
                random_state=config.RANDOM_STATE,
            )
            cols = [f"feature_{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def cargar_clasificacion_lda_qda(dataset: str = "breast_cancer") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga dataset para LDA/QDA. Por defecto Breast Cancer.
    """
    if dataset == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        return data.data, data.target
    elif dataset == "wine":
        data = load_wine(as_frame=True)
        return data.data, data.target
    else:
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=config.RANDOM_STATE,
        )
        cols = [f"feature_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def crear_dataset_sintetico_covarianzas_distintas(
    n_muestras: int = 1500,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea un dataset sintético con covarianzas muy distintas por clase.

    Las tres clases comparten la misma media (0, 0) para aislar el efecto de
    la forma de la covarianza: LDA solo puede usar una frontera lineal y
    prácticamente no separa las clases, mientras que QDA sí puede explotar
    la orientación y escala de cada covarianza. Este es el escenario teórico
    donde QDA supera claramente a LDA.
    """
    rng = np.random.RandomState(random_state)
    n_clases = 3
    n_por_clase = n_muestras // n_clases

    medias = [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])]
    covs = [
        np.array([[3.0, 0.5], [0.5, 0.2]]),       # Elongada en dirección positiva
        np.array([[0.2, -0.5], [-0.5, 3.0]]),     # Elongada en dirección negativa
        np.array([[2.0, 1.8], [1.8, 2.0]]),       # Elongada en diagonal
    ]
    
    X_list, y_list = [], []
    for i in range(n_clases):
        X_c = rng.multivariate_normal(medias[i], covs[i], size=n_por_clase)
        X_list.append(X_c)
        y_list.append(np.full(n_por_clase, i))
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # Mezclar
    indices = rng.permutation(len(X))
    return X[indices], y[indices]
