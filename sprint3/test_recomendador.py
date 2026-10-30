# -*- coding: utf-8 -*-
"""Pruebas del pipeline (Sprint 3 · 6.1). Ejecutar: pytest -q"""
import json
import pytest

rec = pytest.importorskip("recomendador")


@pytest.fixture(scope="module")
def datos():
    try:
        r = rec.obtener()
    except FileNotFoundError as e:  # en CI sin los datos pesados
        pytest.skip(str(e))
    perfil = [p for p in r.listar_perfiles() if p["split"] == "test"][0]["perfil"]
    # Solo prendas base con al menos una combinacion valida: algunas (p. ej. un vestido que ya
    # trae dos estampados) hacen que la regla de doble estampado descarte todas las candidatas.
    prendas = [p for p in r.listar_prendas(n=60)
               if r.recomendar(perfil, p["annotation_id"], top_k=1)["recomendaciones"]][:5]
    if not prendas:
        pytest.fail("Ninguna prenda base produce recomendaciones validas")
    return r, perfil, prendas


def test_devuelve_top_k(datos):
    r, perfil, prendas = datos
    out = r.recomendar(perfil, prendas[0]["annotation_id"], top_k=5)
    assert 1 <= len(out["recomendaciones"]) <= 5


def test_formato_json(datos):
    r, perfil, prendas = datos
    out = r.recomendar(perfil, prendas[0]["annotation_id"], top_k=3)
    json.dumps(out)  # debe ser serializable
    for k in ("perfil", "arquetipo", "contexto", "prenda_base", "recomendaciones"):
        assert k in out
    for x in out["recomendaciones"]:
        assert {"rank", "categoria", "compat", "taste", "final", "reglas_violadas"} <= set(x)


def test_sin_violaciones_de_reglas(datos):
    r, perfil, prendas = datos
    for p in prendas:
        for ocasion, clima in [("oficina", "frio"), ("noche", "entretiempo")]:
            out = r.recomendar(perfil, p["annotation_id"], top_k=10,
                               ocasion=ocasion, clima=clima)
            assert all(x["penalizacion"] == 0 for x in out["recomendaciones"])


def test_ordenado_por_score(datos):
    r, perfil, prendas = datos
    out = r.recomendar(perfil, prendas[0]["annotation_id"], top_k=10)
    f = [x["final"] for x in out["recomendaciones"]]
    assert f == sorted(f, reverse=True)


def test_no_recomienda_categorias_redundantes(datos):
    r, perfil, prendas = datos
    out = r.recomendar(perfil, prendas[0]["annotation_id"], top_k=10)
    base = out["prenda_base"]["categoria"]
    for x in out["recomendaciones"]:
        assert frozenset((base, x["categoria"])) not in rec.PARES_REDUNDANTES
        assert x["categoria"] != base
