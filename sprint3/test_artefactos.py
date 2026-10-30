# -*- coding: utf-8 -*-
"""Comprueba que datos, modelos y reglas estan presentes (Sprint 3 · Diego · MLOps).

- Sin variable: si falta algo, la prueba se OMITE (asi el CI de GitHub, que no tiene los
  datos pesados, no se pone en rojo).
- Con STYLEMATE_EXIGIR_ARTEFACTOS=1: si falta algo, la prueba FALLA. Es la que se usa en la
  prueba final desde cero (contenedor o maquina limpia tras `dvc pull`).
"""
import os

import pytest

EXIGIR = os.environ.get("STYLEMATE_EXIGIR_ARTEFACTOS") == "1"

DATOS = ["instancias_split.csv", "embeddings.npy", "estilos_instancias.npy",
         "perfiles_sinteticos.json", "pares_scored.csv", "preferencias_sinteticas.csv"]
MODELOS = ["compat_head.pt", "taste_head.pt"]


def _modulo():
    if EXIGIR:
        import recomendador  # noqa: PLC0415  (si falla el import, la prueba falla)
        return recomendador
    return pytest.importorskip("recomendador")


def _buscar(nombre, env, obligatorio=True):
    rec = _modulo()
    try:
        ruta = rec._localizar(nombre, env, obligatorio=obligatorio)
    except FileNotFoundError as e:
        if EXIGIR:
            pytest.fail(f"Falta el artefacto '{nombre}'. {e}")
        pytest.skip(f"Falta '{nombre}' (normal en CI sin datos pesados)")
    if ruta is None:
        if EXIGIR:
            pytest.fail(f"Falta '{nombre}'")
        pytest.skip(f"Falta '{nombre}'")
    return ruta


@pytest.mark.parametrize("nombre", DATOS)
def test_dato_presente(nombre):
    assert _buscar(nombre, "STYLEMATE_DATA").stat().st_size > 0


@pytest.mark.parametrize("nombre", MODELOS)
def test_modelo_presente(nombre):
    assert _buscar(nombre, "STYLEMATE_MODELS").stat().st_size > 0


def test_rules_py_del_sprint1():
    # Sin rules.py el sistema usa reglas simplificadas: no es el resultado del Sprint 2.
    ruta = _buscar("E4_reglas_gusto/rules.py", "STYLEMATE_RULES", obligatorio=False)
    assert ruta.stat().st_size > 0


def test_recortes_presentes():
    # Sin recortes la interfaz muestra siluetas «sin recorte» en lugar de fotos.
    assert _buscar("recortes_144.npy", "STYLEMATE_DATA", obligatorio=False).stat().st_size > 0
