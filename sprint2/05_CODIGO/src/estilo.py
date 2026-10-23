# -*- coding: utf-8 -*-
"""Lexico extendido, vectores de estilo de 40 dimensiones, color y perfiles."""

from __future__ import annotations

import importlib.util
from typing import Iterable

import cv2
import numpy as np

import config


def _cargar_rules():
    spec = importlib.util.spec_from_file_location("rules_sprint1", config.RULES_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RULES = _cargar_rules()
LEXICO = RULES.cargar_lexico()
ORDEN = RULES.orden_dimensiones(LEXICO)
DIM = {nombre: i for i, nombre in enumerate(ORDEN)}

CATEGORIA_TERMINOS = {
    "shirt, blouse": ["oficina", "clasico", "entallado", "tela"],
    "top, t-shirt, sweatshirt": ["casual", "diario", "algodon"],
    "sweater": ["punto", "invierno", "casual"],
    "cardigan": ["punto", "entretiempo", "clasico"],
    "jacket": ["entretiempo", "urbano", "cuero"],
    "vest": ["clasico", "oficina", "entallado"],
    "pants": ["oficina", "diario", "recto"],
    "shorts": ["calor", "casual", "verano"],
    "skirt": ["clasico", "entallado", "diario"],
    "coat": ["invierno", "frio", "clasico"],
    "dress": ["fiesta", "elegante", "entallado", "noche"],
    "jumpsuit": ["moderno", "noche", "entallado"],
    "cape": ["invierno", "bohemio", "holgado"],
    "glasses": ["moderno", "diario"],
    "hat": ["verano", "casual", "calor"],
    "headband, head covering, hair accessory": ["casual", "bohemio"],
    "tie": ["formal", "oficina", "clasico"],
    "glove": ["invierno", "frio"],
    "watch": ["clasico", "oficina"],
    "belt": ["clasico", "entallado"],
    "leg warmer": ["invierno", "deportivo"],
    "tights, stockings": ["invierno", "entallado"],
    "sock": ["casual", "deportivo"],
    "shoe": ["diario", "formal", "tela"],
    "bag, wallet": ["diario", "clasico"],
    "scarf": ["invierno", "frio"],
    "umbrella": ["lluvia"],
}

COLOR_TERMINOS = {
    "negro": ["negro", "oscuro", "neutro"],
    "blanco": ["blanco", "neutro"],
    "gris": ["gris", "neutro"],
    "beige": ["beige", "neutro", "calido"],
    "marron": ["calido", "oscuro"],
    "rojo": ["vivo", "calido"],
    "naranja": ["vivo", "calido"],
    "amarillo": ["vivo", "calido"],
    "verde": ["frio", "vivo"],
    "azul": ["frio", "neutro"],
    "morado": ["frio", "vivo"],
    "rosa": ["pastel", "vivo"],
    "": ["neutro"],
}

ATRIBUTO_CLAVES = [
    ("floral", ["floral"]),
    ("stripe", ["rayas"]),
    ("check", ["cuadros"]),
    ("houndstooth", ["cuadros"]),
    ("herringbone", ["cuadros"]),
    ("chevron", ["cuadros"]),
    ("argyle", ["cuadros"]),
    ("dot", ["grafico"]),
    ("geometric", ["grafico"]),
    ("abstract", ["grafico"]),
    ("cartoon", ["grafico"]),
    ("letters", ["grafico"]),
    ("plain", ["liso"]),
    ("animal", ["estampado"]),
    ("leopard", ["estampado"]),
    ("oversized", ["oversize", "holgado"]),
    ("loose", ["holgado"]),
    ("baggy", ["holgado"]),
    ("wide leg", ["holgado"]),
    ("tight", ["ajustado"]),
    ("pencil", ["ajustado"]),
    ("a-line", ["holgado"]),
    ("flare", ["holgado"]),
    ("circle", ["holgado"]),
    ("mini", ["fiesta"]),
    ("micro", ["fiesta"]),
    ("maxi", ["elegante"]),
    ("floor", ["elegante"]),
    ("sleeveless", ["calor"]),
    ("v-neck", ["noche"]),
    ("plunging", ["noche"]),
    ("off-the-shoulder", ["noche"]),
    ("zip-up", ["urbano"]),
    ("biker", ["urbano"]),
    ("denim", ["denim"]),
    ("leather", ["cuero"]),
    ("knit", ["punto"]),
    ("lace", ["bohemio"]),
    ("sequin", ["fiesta"]),
    ("bead", ["fiesta"]),
    ("glitter", ["fiesta"]),
    ("ruffle", ["bohemio"]),
    ("fringe", ["bohemio"]),
    ("bow", ["bohemio"]),
    ("pleat", ["clasico"]),
    ("ruched", ["entallado"]),
    ("classic", ["clasico"]),
    ("sport", ["deportivo"]),
    ("formal", ["formal"]),
    ("casual", ["casual"]),
]


def _norm_max(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    m = float(v.max()) if v.size else 0.0
    return v / m if m > 0 else v


def vector_terminos(terminos: Iterable[str]) -> np.ndarray:
    return RULES.vector_desde_terminos(list(terminos), LEXICO)


def terminos_atributo(nombre: str) -> list[str]:
    n = str(nombre).lower()
    out = []
    for clave, terms in ATRIBUTO_CLAVES:
        if clave in n:
            out.extend(terms)
    return out


def vector_prenda(categoria: str, atributos: Iterable[str],
                  color: str = "") -> np.ndarray:
    terminos = list(CATEGORIA_TERMINOS.get(categoria, ["diario"]))
    for a in atributos:
        terminos.extend(terminos_atributo(a))
    terminos.extend(COLOR_TERMINOS.get(str(color).lower(), ["neutro"]))
    v = vector_terminos(terminos)
    # La senal de categoria siempre esta presente.
    if float(v.sum()) == 0:
        v = vector_terminos(["diario", "casual"])
    return v.astype(np.float32)


def vector_par(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    return (0.5 * (va + vb)).astype(np.float32)


def coseno(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# Color dominante desde la mascara de segmentation
# --------------------------------------------------------------------------- #
def _rle_decode(counts, shape) -> np.ndarray:
    if isinstance(counts, str):
        s = [int(x) for x in counts.split()]
    else:
        s = [int(x) for x in counts]
    img = np.zeros(int(shape[0]) * int(shape[1]), dtype=np.uint8)
    pos = 0
    for i in range(0, len(s), 2):
        start = s[i]
        length = s[i + 1] if i + 1 < len(s) else 0
        if start:
            pos += start
        if length:
            img[pos:pos + length] = 1
            pos += length
    return img.reshape((int(shape[1]), int(shape[0]))).T


def mascara_desde_segmentacion(seg, alto: int, ancho: int) -> np.ndarray:
    mask = np.zeros((alto, ancho), dtype=np.uint8)
    if isinstance(seg, list):
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts[:, 0] = np.clip(pts[:, 0], 0, ancho - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, alto - 1)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    elif isinstance(seg, dict) and "counts" in seg:
        try:
            m = _rle_decode(seg["counts"], (alto, ancho))
            if m.shape == (alto, ancho):
                mask = m.astype(np.uint8)
        except Exception:
            pass
    return mask


def _lab_paleta() -> tuple[list[str], np.ndarray, dict[str, tuple[int, int, int]]]:
    nombres, labs = [], []
    for nombre, rgb in config.PALETA_COLORES.items():
        arr = np.uint8([[rgb]])
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        nombres.append(nombre)
        labs.append(lab)
    return nombres, np.vstack(labs), dict(config.PALETA_COLORES)


PALETA_NOMBRES, PALETA_LAB, PALETA_RGB = _lab_paleta()


def color_desde_lab(lab: np.ndarray) -> str:
    d = np.linalg.norm(PALETA_LAB - lab.reshape(1, 3), axis=1)
    return PALETA_NOMBRES[int(np.argmin(d))]


def extraer_color(imagen_rgb: np.ndarray, seg, bbox) -> tuple[str, list[float], str]:
    alto, ancho = imagen_rgb.shape[:2]
    mask = mascara_desde_segmentacion(seg, alto, ancho)
    x, y, w, h = [int(round(float(v))) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(ancho, x + max(1, w)), min(alto, y + max(1, h))
    if x1 <= x0 or y1 <= y0:
        return "gris", [0.0, 0.0, 0.0], "#808080"
    crop = imagen_rgb[y0:y1, x0:x1]
    mcrop = mask[y0:y1, x0:x1]
    if int(mcrop.sum()) < 20:
        mcrop = np.ones((y1 - y0, x1 - x0), dtype=np.uint8)
    pix = crop[mcrop > 0].reshape(-1, 3)
    if len(pix) == 0:
        pix = crop.reshape(-1, 3)
    if len(pix) > 1200:
        pix = pix[:: max(1, len(pix) // 1200)]
    pix = pix.astype(np.uint8)
    lab = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    lab = lab[(lab[:, 0] > 8) & (lab[:, 0] < 99)]
    if len(lab) < 5:
        lab = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    k = int(min(3, max(1, len(lab))))
    if k == 1:
        centro = lab.mean(axis=0)
        nombre = color_desde_lab(centro)
    else:
        data = lab.astype(np.float32)
        _, etiquetas, centros = cv2.kmeans(
            data, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            1, cv2.KMEANS_RANDOM_CENTERS,
        )
        etiquetas = etiquetas.ravel()
        idx = int(np.argmax(np.bincount(etiquetas, minlength=k)))
        centro = centros[idx]
        nombre = color_desde_lab(centro)
    rgb = PALETA_RGB.get(nombre, (128, 128, 128))
    hexa = "#{:02x}{:02x}{:02x}".format(*rgb)
    return nombre, [float(round(v, 2)) for v in centro], hexa


# --------------------------------------------------------------------------- #
# Perfiles sinteticos y afinidad
# --------------------------------------------------------------------------- #
ARQUETIPOS = {
    "casual": ["casual", "diario", "liso", "holgado", "neutro", "algodon"],
    "formal": ["formal", "oficina", "clasico", "entallado", "neutro", "tela"],
    "deportivo": ["deportivo", "deporte", "tecnico", "holgado", "frio", "diario"],
    "elegante": ["elegante", "fiesta", "noche", "entallado", "oscuro", "clasico"],
    "bohemio": ["bohemio", "floral", "calor", "holgado", "vivo", "viaje"],
}


def perfil_sintetico(arquetipo: str, ruido: float, rng: np.random.Generator) -> np.ndarray:
    base = vector_terminos(ARQUETIPOS.get(arquetipo, ARQUETIPOS["casual"]))
    if float(base.sum()) == 0:
        base = np.ones(DIM.__len__(), dtype=float)
    v = base + rng.normal(0.0, ruido, size=base.shape)
    v = np.clip(v, 0.0, None)
    return _norm_max(v).astype(np.float32)


def generar_perfiles(n: int, split: str, ruido: float,
                     seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    tipos = list(ARQUETIPOS)
    out = []
    for i in range(n):
        arq = tipos[i % len(tipos)]
        v = perfil_sintetico(arq, ruido, rng)
        out.append({
            "profile_id": f"{split}_{i:04d}",
            "split": split,
            "arquetipo": arq,
            "ruido": float(ruido),
            "vector": [float(round(x, 5)) for x in v],
            "dimensiones_activas": [ORDEN[j] for j in np.nonzero(v > 0.05)[0]],
        })
    return out


def afinidad_perfil_par(perfil: np.ndarray, va: np.ndarray, vb: np.ndarray) -> float:
    vp = vector_par(va, vb)
    c = coseno(perfil, vp)
    # Cobertura: fraccion de dimensiones activas del perfil presentes en el par.
    activas = np.nonzero(perfil > 0.15)[0]
    if len(activas) == 0:
        return c
    cobertura = float(np.mean(vp[activas] > 0.05))
    return float(0.75 * c + 0.25 * cobertura)
