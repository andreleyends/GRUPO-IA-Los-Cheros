# -*- coding: utf-8 -*-
"""Utilidades comunes del Sprint 2."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def json_dump(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def json_load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for p in [config.PROCESSED, config.CACHE, config.MODELS, config.REPORTS,
              config.FIGURAS, config.METRICAS, config.DOCS, config.SCRUM]:
        p.mkdir(parents=True, exist_ok=True)


def muestrear_estratificado(df: pd.DataFrame, n: int, col: str,
                            seed: int = config.SEED) -> pd.DataFrame:
    """Muestrea n filas respetando la distribucion de col, con minimo 1."""
    if n >= len(df):
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    counts = df[col].fillna("sin_dato").value_counts()
    total = int(counts.sum())
    alloc = {}
    for k, v in counts.items():
        cuota = max(1, int(round(n * v / total))) if v >= 3 else 1
        alloc[k] = min(int(v), cuota)

    def ajustar(delta):
        keys = list(alloc)
        rng.shuffle(keys)
        for k in keys:
            if delta > 0 and alloc[k] < counts[k]:
                alloc[k] += 1
                delta -= 1
            elif delta < 0 and alloc[k] > 1:
                alloc[k] -= 1
                delta += 1
            if delta == 0:
                break
        return delta

    delta = n - sum(alloc.values())
    while delta != 0:
        restante = ajustar(delta)
        if restante == delta:
            break
        delta = restante

    partes = []
    for i, (k, a) in enumerate(alloc.items()):
        sub = df[df[col].fillna("sin_dato") == k]
        if a > 0 and len(sub):
            partes.append(sub.sample(n=min(a, len(sub)),
                                     random_state=seed + i))
    out = pd.concat(partes, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def configurar_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })
    return plt


def guardar_figura(fig, nombre: str) -> Path:
    path = config.FIGURAS / nombre
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def ahora() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt_pct(x: float) -> str:
    return f"{100 * float(x):.1f} %"


def fmt_num(x) -> str:
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)
