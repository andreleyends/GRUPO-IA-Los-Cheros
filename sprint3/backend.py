# -*- coding: utf-8 -*-
"""
StyleMate · Sprint 3 · Selector de backend para la interfaz (Andre)

La interfaz (app.py) solo habla con este modulo. Aqui se decide si se usa:
  - el recomendador REAL de Marcela (recomendador.py + modelos + datos), o
  - el MOCK (mock_recomendador.py), con el mismo formato JSON.

Variable de entorno STYLEMATE_BACKEND:
  auto (por defecto) -> intenta el real; si falta algo, cae al mock y guarda el motivo
  real               -> exige el real; si falla, lanza el error (para la demo final)
  mock               -> fuerza el mock
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
if str(AQUI) not in sys.path:
    sys.path.insert(0, str(AQUI))

import mock_recomendador as _mock  # noqa: E402

OCASIONES = _mock.OCASIONES
CLIMAS = _mock.CLIMAS

_real = None          # modulo recomendador si cargo bien
MODO = "mock"         # "real" | "mock"
MOTIVO_MOCK = ""      # por que se uso el mock (vacio si es real)


def _intentar_real():
    """Importa recomendador.py y carga modelos/datos. Devuelve el modulo o lanza error."""
    import recomendador  # noqa: PLC0415  (puede fallar si falta torch)
    recomendador.obtener()  # carga datos y modelos ahora, no en el primer clic
    return recomendador


def _inicializar():
    global _real, MODO, MOTIVO_MOCK
    pedido = os.environ.get("STYLEMATE_BACKEND", "auto").strip().lower()
    if pedido == "mock":
        MODO, MOTIVO_MOCK = "mock", "forzado con STYLEMATE_BACKEND=mock"
        return
    try:
        _real = _intentar_real()
        MODO, MOTIVO_MOCK = "real", ""
    except Exception as e:  # noqa: BLE001
        if pedido == "real":
            raise
        MODO = "mock"
        MOTIVO_MOCK = f"{type(e).__name__}: {e}"


_inicializar()


def _b():
    return _real if MODO == "real" else _mock


def listar_perfiles() -> list[dict]:
    return _b().listar_perfiles()


def listar_prendas(categoria=None, n=24) -> list[dict]:
    return _b().listar_prendas(categoria, n)


def categorias() -> list[str]:
    """Categorias que existen en el catalogo (para el filtro de la prenda base)."""
    if MODO == "real":
        r = _real.obtener()
        return sorted({str(x) for x in r.nombres[r.catalogo]})
    return _mock.categorias()


def recomendar(perfil, prenda_base, top_k=5, **kw) -> dict:
    return _b().recomendar(perfil, prenda_base, top_k, **kw)


def recorte(pos: int, categoria: str = "", color: str = ""):
    """Imagen 144x144 de la prenda.

    Real: recorte de Fashionpedia (recortes_144.npy). Si ese archivo no esta, devuelve una
    silueta de la categoria en su color, marcada «sin recorte», para que la interfaz siga util.
    """
    if MODO == "real":
        arr = _real.obtener().recorte(pos)
        return arr if arr is not None else _mock.dibujar(categoria, color, "sin recorte")
    return _mock.recorte(pos)
