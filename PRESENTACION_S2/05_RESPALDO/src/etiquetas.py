# -*- coding: utf-8 -*-
"""Etiquetas de las cabezas de atributos del CNN."""

from __future__ import annotations

import pandas as pd

import config


def parse_attribute_ids(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, float) and pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    s = s.replace(",", " ").replace(";", " ")
    return [int(x) for x in s.split() if x.strip().lstrip("-").isdigit()]


def etiquetas_cabezas(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    attrs = df["attribute_ids"].apply(parse_attribute_ids)
    for clave, cfg in config.HEAD_ATTRS.items():
        mapa = {aid: i for i, aid in enumerate(cfg["ids"])}
        vals = []
        for lista in attrs:
            etiqueta = -100
            for aid in lista:
                if aid in mapa:
                    etiqueta = mapa[aid]
                    break
            vals.append(etiqueta)
        out[f"y_{clave}"] = vals
    return out
