# -*- coding: utf-8 -*-
"""Pruebas de la interfaz y del contrato JSON (Sprint 3 · Andre). Ejecutar: pytest -q"""
import json
import os
from pathlib import Path

import pytest

AQUI = Path(__file__).resolve().parent
CLAVES_SALIDA = {"perfil", "arquetipo", "contexto", "prenda_base", "recomendaciones",
                 "descartadas_por_reglas"}
CLAVES_PRENDA = {"annotation_id", "categoria", "color", "pos", "image_id", "file_name"}
CLAVES_REC = CLAVES_PRENDA | {"compat", "taste", "penalizacion", "final",
                              "reglas_violadas", "rank"}


def _backends():
    import mock_recomendador as mock
    lista = [pytest.param(mock, id="mock")]
    try:
        import recomendador as real
        real.obtener()
        lista.append(pytest.param(real, id="real"))
    except Exception as e:  # noqa: BLE001  (sin torch o sin datos: se omite)
        lista.append(pytest.param(None, id="real", marks=pytest.mark.skip(reason=str(e)[:80])))
    return lista


@pytest.fixture(params=_backends())
def be(request):
    return request.param


def _un_caso(be):
    perfil = [p for p in be.listar_perfiles() if p["split"] == "test"][0]["perfil"]
    # Base con al menos una combinacion valida (algunas prendas quedan sin candidatas por reglas)
    prenda = next(p for p in be.listar_prendas(n=60)
                  if be.recomendar(perfil, p["annotation_id"], top_k=1)["recomendaciones"])
    return perfil, prenda


def test_contrato_json(be):
    perfil, prenda = _un_caso(be)
    out = be.recomendar(perfil, prenda["annotation_id"], top_k=5)
    json.dumps(out)
    assert CLAVES_SALIDA <= set(out)
    assert {"ocasion", "clima"} <= set(out["contexto"])
    assert CLAVES_PRENDA <= set(out["prenda_base"])
    assert 1 <= len(out["recomendaciones"]) <= 5
    for r in out["recomendaciones"]:
        assert CLAVES_REC <= set(r)


def test_ranking_ordenado_y_numerado(be):
    perfil, prenda = _un_caso(be)
    out = be.recomendar(perfil, prenda["annotation_id"], top_k=10)
    f = [r["final"] for r in out["recomendaciones"]]
    assert f == sorted(f, reverse=True)
    assert [r["rank"] for r in out["recomendaciones"]] == list(range(1, len(f) + 1))


def test_sin_violaciones_con_filtro(be):
    perfil, prenda = _un_caso(be)
    out = be.recomendar(perfil, prenda["annotation_id"], top_k=10,
                        ocasion="oficina", clima="frio")
    assert all(r["penalizacion"] == 0 and not r["reglas_violadas"]
               for r in out["recomendaciones"])


def test_contexto_se_respeta(be):
    perfil, prenda = _un_caso(be)
    out = be.recomendar(perfil, prenda["annotation_id"], ocasion="noche", clima="calor")
    assert out["contexto"] == {"ocasion": "noche", "clima": "calor"}


def test_prenda_inexistente_da_error(be):
    perfil, _ = _un_caso(be)
    with pytest.raises(KeyError):
        be.recomendar(perfil, -1)


def test_mock_y_real_tienen_las_mismas_claves():
    """El mock debe poder reemplazarse por el real sin tocar la interfaz."""
    real = pytest.importorskip("recomendador")
    try:
        real.obtener()
    except Exception as e:  # noqa: BLE001
        pytest.skip(str(e)[:80])
    import mock_recomendador as mock
    p_r = [p for p in real.listar_perfiles() if p["split"] == "test"][0]["perfil"]
    base_r = next(p for p in real.listar_prendas(n=60)
                  if real.recomendar(p_r, p["annotation_id"], top_k=1)["recomendaciones"])
    a = real.recomendar(p_r, base_r["annotation_id"], top_k=3)
    b = mock.recomendar("test_0000", mock.listar_prendas(n=1)[0]["annotation_id"], top_k=3)
    assert set(a) == set(b)
    assert set(a["recomendaciones"][0]) == set(b["recomendaciones"][0])


# ----------------------------------------------------------------- interfaz
@pytest.fixture
def app_mock(monkeypatch):
    """Arranca app.py con el mock (no depende de torch ni de los datos pesados)."""
    monkeypatch.setenv("STYLEMATE_BACKEND", "mock")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(AQUI / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_app_arranca_con_aviso_de_mock(app_mock):
    assert any("MODO DEMO" in w.value for w in app_mock.warning)
    assert any("Recomendar" in i.value for i in app_mock.info)


def test_app_recomienda_y_muestra_looks(app_mock):
    app_mock.button[0].click().run()
    assert not app_mock.exception
    assert any("Looks recomendados" in s.value for s in app_mock.subheader)
    assert len(app_mock.metric) >= 2          # compat. y gusto de cada look


def test_app_sin_resultados_muestra_aviso(app_mock):
    # shorts + frio + oficina deja sin candidatas validas si la base es shorts
    cat = next(s for s in app_mock.selectbox if s.label == "Categoría")
    cat.set_value("shorts")
    app_mock.run()
    next(s for s in app_mock.selectbox if s.label == "Clima").set_value("frio")
    app_mock.run()
    app_mock.button[0].click().run()
    assert not app_mock.exception
    assert any("No hay combinaciones" in w.value for w in app_mock.warning)
