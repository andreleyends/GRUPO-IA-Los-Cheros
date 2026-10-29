# -*- coding: utf-8 -*-
"""
StyleMate · Sprint 3 · Mock del recomendador (Andre)

Devuelve EXACTAMENTE el mismo formato JSON que recomendador.recomendar() de Marcela,
pero con datos inventados. Sirve para desarrollar y probar la interfaz sin los
modelos ni los datos pesados. Los puntajes NO son de ningun modelo.

API identica a recomendador.py:
    recomendar(perfil, prenda_base, top_k=5, ocasion=None, clima=None,
               una_por_categoria=True, solo_validas=True) -> dict
    listar_prendas(categoria=None, n=24) -> list[dict]
    listar_perfiles() -> list[dict]
    recorte(pos) -> ndarray RGB 144x144
    categorias() -> list[str]
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageDraw

OCASIONES = ["diario", "oficina", "deporte", "noche", "viaje"]
CLIMAS = ["frio", "entretiempo", "calor"]

CONTEXTO = {  # igual que recomendador.py
    "casual": {"clima": "entretiempo", "ocasion": "diario"},
    "formal": {"clima": "entretiempo", "ocasion": "oficina"},
    "deportivo": {"clima": "entretiempo", "ocasion": "deporte"},
    "elegante": {"clima": "entretiempo", "ocasion": "noche"},
    "bohemio": {"clima": "calor", "ocasion": "viaje"},
}
PARES_REDUNDANTES = {frozenset(p) for p in [
    ("dress", "skirt"), ("dress", "pants"), ("dress", "shorts"), ("dress", "jumpsuit"),
    ("skirt", "pants"), ("skirt", "shorts"), ("pants", "shorts"),
    ("jumpsuit", "pants"), ("jumpsuit", "skirt"), ("jumpsuit", "shorts"),
]}
CATEGORIAS = ["shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "jacket", "coat",
              "pants", "shorts", "skirt", "dress", "shoe", "bag, wallet", "belt", "hat", "scarf"]
COLORES = {  # paleta del Sprint 2 (sprint2/02_DATASET/color_paleta.json)
    "negro": (20, 20, 20), "gris": (128, 128, 128), "blanco": (245, 245, 245),
    "beige": (222, 202, 170), "marron": (120, 80, 50), "rojo": (200, 30, 40),
    "naranja": (235, 130, 40), "amarillo": (240, 210, 60), "verde": (50, 140, 70),
    "azul": (40, 80, 180), "morado": (120, 50, 150), "rosa": (230, 130, 170),
}
COLORES_VIVOS = {"rojo", "naranja", "amarillo", "verde", "morado", "rosa"}
_ARQ = ["casual", "formal", "deportivo", "elegante", "bohemio"]
N_PRENDAS = 240


def _h(*partes) -> float:
    """Numero estable en [0,1) a partir de los argumentos (reproducible)."""
    s = "|".join(str(p) for p in partes).encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16) / 0xFFFFFFFF


def _catalogo() -> list[dict]:
    nombres_col = list(COLORES)
    out = []
    for pos in range(N_PRENDAS):
        out.append({
            "annotation_id": 900000 + pos,
            "categoria": CATEGORIAS[pos % len(CATEGORIAS)],
            "color": nombres_col[int(_h("c", pos) * len(nombres_col))],
            "pos": pos,
            "image_id": 5000 + pos // 2,
            "file_name": f"mock_{5000 + pos // 2:06d}.jpg",
        })
    return out


_CAT = _catalogo()
_POR_ANN = {p["annotation_id"]: p for p in _CAT}
_PERFILES = [{"perfil": f"test_{i:04d}", "arquetipo": _ARQ[i % 5], "split": "test"}
             for i in range(10)]


def categorias() -> list[str]:
    return sorted({p["categoria"] for p in _CAT})


def listar_perfiles() -> list[dict]:
    return [dict(p) for p in _PERFILES]


def listar_prendas(categoria=None, n=24) -> list[dict]:
    idx = [p for p in _CAT if categoria in (None, "", p["categoria"])]
    orden = sorted(idx, key=lambda p: _h("orden", p["pos"]))
    return [dict(p) for p in orden[:n]]


def dibujar(categoria: str, color: str, marca: str = "") -> np.ndarray:
    """Silueta 144x144 de la categoria en el color indicado (placeholder cuando no hay recorte)."""
    rgb = COLORES.get(color, (150, 150, 150))
    img = Image.new("RGB", (144, 144), (238, 236, 232))
    d = ImageDraw.Draw(img)
    c = categoria
    osc = tuple(int(v * 0.6) for v in rgb)
    if c in ("pants", "tights, stockings", "leg warmer"):
        d.polygon([(48, 20), (96, 20), (104, 128), (78, 128), (72, 62), (66, 128), (40, 128)], fill=rgb)
    elif c == "shorts":
        d.polygon([(48, 38), (96, 38), (102, 88), (78, 88), (72, 64), (66, 88), (42, 88)], fill=rgb)
    elif c == "skirt":
        d.polygon([(54, 30), (90, 30), (112, 114), (32, 114)], fill=rgb)
    elif c in ("dress", "jumpsuit"):
        d.polygon([(56, 14), (88, 14), (92, 52), (116, 126), (28, 126), (52, 52)], fill=rgb)
    elif c == "shoe":
        d.rounded_rectangle([26, 78, 118, 108], 14, fill=rgb)
        d.rectangle([26, 100, 118, 112], fill=osc)
    elif c == "bag, wallet":
        d.rounded_rectangle([34, 56, 110, 118], 8, fill=rgb)
        d.arc([48, 22, 96, 84], 180, 360, fill=osc, width=6)
    elif c in ("belt", "tie"):
        d.rectangle([14, 66, 130, 80], fill=rgb)
        d.rectangle([66, 62, 80, 84], outline=(190, 170, 90), width=3)
    elif c in ("hat", "headband, head covering, hair accessory"):
        d.ellipse([22, 84, 122, 110], fill=rgb)
        d.pieslice([44, 38, 100, 100], 180, 360, fill=rgb)
    elif c in ("scarf", "cape"):
        d.rounded_rectangle([40, 24, 104, 56], 12, fill=rgb)
        d.rectangle([76, 50, 100, 122], fill=rgb)
    elif c == "glasses":
        d.ellipse([22, 56, 66, 94], outline=rgb, width=6)
        d.ellipse([78, 56, 122, 94], outline=rgb, width=6)
        d.line([(66, 70), (78, 70)], fill=rgb, width=5)
    else:  # prendas superiores y resto
        d.polygon([(52, 20), (92, 20), (126, 54), (110, 72), (98, 60), (98, 122), (46, 122),
                   (46, 60), (34, 72), (18, 54)], fill=rgb)
        if c in ("jacket", "coat", "cardigan", "vest"):
            d.line([(72, 22), (72, 122)], fill=osc, width=3)
    if marca:
        d.text((6, 4), marca, fill=(110, 110, 110))
    return np.asarray(img)


def recorte(pos: int, categoria: str = "", color: str = "") -> np.ndarray:
    """Imagen del mock (siluetas de color). Marca MOCK."""
    p = _CAT[int(pos) % N_PRENDAS]
    return dibujar(p["categoria"], p["color"], "MOCK")


def _reglas(cat_a, col_a, cat_b, col_b, ctx) -> list[str]:
    """Version reducida de las reglas del Sprint 1 (solo las que se pueden evaluar sin atributos)."""
    v = []
    cats = {cat_a, cat_b}
    if ctx["clima"] == "frio" and "shorts" in cats:
        v.append("shorts_con_frio")
    if ctx["ocasion"] == "oficina" and "shorts" in cats:
        v.append("deportivo_en_oficina")
    if int(col_a in COLORES_VIVOS) + int(col_b in COLORES_VIVOS) >= 2:
        v.append("exceso_colores_vivos")
    return v


def recomendar(perfil, prenda_base, top_k=5, ocasion=None, clima=None,
               una_por_categoria=True, solo_validas=True) -> dict:
    perf = next((p for p in _PERFILES if p["perfil"] == perfil), None)
    if perf is None:
        raise KeyError(f"Perfil desconocido: {perfil}")
    base = _POR_ANN.get(int(prenda_base))
    if base is None:
        raise KeyError(f"prenda_base {prenda_base} no existe (usa annotation_id)")

    ctx = dict(CONTEXTO[perf["arquetipo"]])
    if ocasion:
        ctx["ocasion"] = ocasion
    if clima:
        ctx["clima"] = clima

    filas, descartadas = [], 0
    for c in _CAT:
        if c["annotation_id"] == base["annotation_id"] or c["categoria"] == base["categoria"]:
            continue
        if frozenset((base["categoria"], c["categoria"])) in PARES_REDUNDANTES:
            continue
        compat = 0.30 + 0.65 * _h("compat", base["annotation_id"], c["annotation_id"])
        taste = min(0.99, 0.55 * compat + 0.45 * _h("taste", perfil, c["annotation_id"]))
        violadas = _reglas(base["categoria"], base["color"], c["categoria"], c["color"], ctx)
        pen = float(len(violadas))
        if pen > 0 and solo_validas:
            descartadas += 1
            continue
        filas.append({**c, "compat": round(compat, 4), "taste": round(taste, 4),
                      "penalizacion": round(pen, 3), "final": round(taste - 0.5 * pen, 4),
                      "reglas_violadas": violadas})
    filas.sort(key=lambda r: r["final"], reverse=True)
    if una_por_categoria:
        vistos, unicas = set(), []
        for r in filas:
            if r["categoria"] not in vistos:
                vistos.add(r["categoria"])
                unicas.append(r)
        filas = unicas
    filas = filas[:top_k]
    for k, r in enumerate(filas, 1):
        r["rank"] = k
    return {"perfil": perfil, "arquetipo": perf["arquetipo"],
            "contexto": {"ocasion": ctx["ocasion"], "clima": ctx["clima"]},
            "prenda_base": dict(base), "recomendaciones": filas,
            "descartadas_por_reglas": int(descartadas)}
