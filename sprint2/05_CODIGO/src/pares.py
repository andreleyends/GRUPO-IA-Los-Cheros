# -*- coding: utf-8 -*-
"""Reglas de generacion de pares de compatibilidad (2.5)."""

from __future__ import annotations

import itertools

import pandas as pd

import config


def es_redundante(cat_a: str, cat_b: str) -> bool:
    return frozenset({str(cat_a), str(cat_b)}) in config.PARES_REDUNDANTES


def positivos_de_imagen(sub: pd.DataFrame):
    """Pares de instancias principales de categorias distintas en una imagen."""
    sub = sub[sub["es_principal"]]
    filas = list(sub.itertuples())
    for a, b in itertools.combinations(filas, 2):
        if int(a.category_id) == int(b.category_id):
            continue
        if es_redundante(a.category_name, b.category_name):
            continue
        # Orden canonico por annotation_id para reproducibilidad.
        if int(a.annotation_id) <= int(b.annotation_id):
            yield a, b
        else:
            yield b, a


def contar_positivos(df: pd.DataFrame, split_map: dict[int, str]) -> dict:
    out = {"train": 0, "val": 0, "test": 0}
    for image_id, sub in df[df["es_principal"]].groupby("image_id"):
        split = split_map.get(int(image_id))
        if split is None:
            continue
        out[split] += sum(1 for _ in positivos_de_imagen(sub))
    return out
