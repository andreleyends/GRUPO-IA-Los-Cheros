"""
Modelos Naive Bayes: MultinomialNB y ComplementNB.

Notas: Para MultinomialNB se usan conteos o TF-IDF (frecuencias no negativas).
NO se aplica StandardScaler (el modelo requiere conteos o frecuencias no negativas).
ComplementNB es más robusto ante datos desbalanceados.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


class VectorizadorSeleccionable(BaseEstimator, TransformerMixin):
    """
    Vectorizador configurable que permite alternar entre CountVectorizer y
    TfidfVectorizer dentro de GridSearchCV.
    """

    def __init__(
        self,
        tipo: str = "tfidf",
        ngram_range: tuple = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
    ) -> None:
        self.tipo = tipo
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

    def _crear(self):
        if self.tipo == "count":
            return CountVectorizer(
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                strip_accents="unicode",
                lowercase=True,
            )
        if self.tipo == "tfidf":
            return TfidfVectorizer(
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                strip_accents="unicode",
                lowercase=True,
            )
        raise ValueError("tipo debe ser 'count' o 'tfidf'")

    def fit(self, X, y=None):
        self.vectorizador_ = self._crear()
        self.vectorizador_.fit(X, y)
        return self

    def transform(self, X):
        return self.vectorizador_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.vectorizador_.get_feature_names_out(input_features)


def crear_pipeline_nb(
    tipo_vectorizador: str = "tfidf",
    clasificador: str = "multinomial",
    alpha: float = 1.0,
    fit_prior: bool = True,
    norm: bool = True,
    ngram_range: tuple = (1, 1),
    min_df: int = 1,
    max_df: float = 1.0,
) -> Pipeline:
    """
    Crea pipeline para Naive Bayes con vectorización.

    Parámetros
    ----------
    tipo_vectorizador : str
        'count' (conteos) o 'tfidf' (frecuencia inversa de documento).
    clasificador : str
        'multinomial' o 'complement'.
    alpha : float
        Suavizado de Laplace/Lidstone.
    fit_prior : bool
        Si se aprenden los priors de clase.
    norm : bool
        Solo para ComplementNB: normaliza los pesos.
    """
    vectorizador = VectorizadorSeleccionable(
        tipo=tipo_vectorizador,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )

    if clasificador == "multinomial":
        clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    elif clasificador == "complement":
        clf = ComplementNB(alpha=alpha, fit_prior=fit_prior, norm=norm)
    else:
        raise ValueError("clasificador debe ser 'multinomial' o 'complement'")

    return Pipeline([
        ("vectorizer", vectorizador),
        ("clf", clf),
    ])


def obtener_palabras_informativas(
    pipeline: Pipeline,
    clase_idx: int,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """
    Obtiene las palabras más informativas para una clase.

    Para MultinomialNB se usa feature_log_prob_ (log P(w|y)).
    Para ComplementNB feature_log_prob_ representa los pesos del complemento
    (mayor valor = mayor evidencia en contra del complemento); igualmente se
    ordenan de mayor a menor como indicador de importancia.
    """
    vectorizer = pipeline.named_steps["vectorizer"]
    clf = pipeline.named_steps["clf"]

    feature_names = vectorizer.get_feature_names_out()
    if hasattr(clf, "feature_log_prob_"):
        log_probs = clf.feature_log_prob_[clase_idx]
        top_indices = np.argsort(log_probs)[::-1][:top_n]
        return [(str(feature_names[i]), float(log_probs[i])) for i in top_indices]
    return []
