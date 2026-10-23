# -*- coding: utf-8 -*-
"""Configuracion central del Sprint 2 (escenario B: 4.000 imagenes)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPRINT1 = ROOT.parent / "sprint1"
IDX = SPRINT1 / "salidas" / "indices"
RAW = SPRINT1 / "datos" / "raw"
IMAGENES = SPRINT1 / "datos" / "imagenes" / "train"
LEXICO = SPRINT1 / "E4_reglas_gusto" / "lexico_moda.json"
RULES_PY = SPRINT1 / "E4_reglas_gusto" / "rules.py"

PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURAS = REPORTS / "figuras"
METRICAS = REPORTS / "metricas"
DOCS = ROOT / "docs"
SCRUM = ROOT / "scrum"

SEED = 42
IMG_SIZE = 144
N_IMAGENES_OBJETIVO = 4000
N_POSITIVAS_MUESTRA = 3400
N_SINGLE_MUESTRA = 600
MIN_AREA = 256

CATEGORIAS_PRINCIPALES = list(range(27))
CATEGORIAS_PARTES = list(range(27, 46))
SUPERCATEGORIAS_PARTE = {"garment parts", "closures", "decorations"}

# Pares de categorias que no deben usarse como positivos por ser redundantes.
PARES_REDUNDANTES = {
    frozenset({"dress", "skirt"}),
    frozenset({"dress", "pants"}),
    frozenset({"dress", "shorts"}),
    frozenset({"dress", "jumpsuit"}),
    frozenset({"skirt", "pants"}),
    frozenset({"skirt", "shorts"}),
    frozenset({"pants", "shorts"}),
    frozenset({"jumpsuit", "pants"}),
    frozenset({"jumpsuit", "skirt"}),
    frozenset({"jumpsuit", "shorts"}),
}

# Etiquetas de compatibilidad (2.5). En el subconjunto de 4k se calibra a 1:2.
NEG_RATIO = 2
HARD_NEG_FRAC = 0.30
MIN_POS_CATPAIR = 20
MAX_SINTETICOS_POR_PAR = 40

# Preferencias sinteticas (2.5). Perfiles disjuntos train/val/test.
N_PROFILES_TRAIN = 120
N_PROFILES_VAL = 30
N_PROFILES_TEST = 30
NOISE_TRAIN = 0.10
NOISE_VAL = 0.15
NOISE_TEST = 0.20
PREF_POS_POR_PERFIL = 250
PREF_NEG_POR_PERFIL = 250
N_ARCHETIPOS = 5

# Modelos.
BATCH_SIZE = 64
CNN_EPOCHS = 15
CNN_LR = 1e-3
CNN_PATIENCE = 5
COMPAT_EPOCHS = 20
COMPAT_LR = 1e-3
TASTE_EPOCHS = 20
TASTE_LR = 1e-3
EMBED_DIM = 128

# Cabezas de atributos: IDs de Fashionpedia seleccionados por el diseno.
HEAD_ATTRS = {
    "silueta": {
        "nombre": "Silueta y ajuste",
        "ids": [114, 115, 116, 118, 128, 129, 135, 136, 137, 138],
        "nombres": [
            "asymmetrical", "symmetrical", "peplum", "flare", "straight",
            "a-line", "tight (fit)", "regular (fit)", "loose (fit)", "oversized",
        ],
    },
    "patron": {
        "nombre": "Patron textil",
        "ids": [317, 318, 321, 322, 323, 325, 326, 328, 331],
        "nombres": [
            "plain", "abstract", "camouflage", "check", "dot", "floral",
            "geometric", "stripe", "chevron",
        ],
    },
    "largo": {
        "nombre": "Largo",
        "ids": [149, 150, 151, 153, 154, 155, 156],
        "nombres": [
            "mini", "above-the-knee", "knee", "midi", "maxi", "floor", "sleeveless",
        ],
    },
    "detalle": {
        "nombre": "Acabado / detalle",
        "ids": [297, 300, 302, 304, 225, 226, 229, 183],
        "nombres": [
            "distressed", "frayed", "ruched", "pleat", "single breasted",
            "double breasted", "zip-up", "v-neck",
        ],
    },
}

PALETA_COLORES = {
    "negro": (20, 20, 20),
    "gris": (128, 128, 128),
    "blanco": (245, 245, 245),
    "beige": (222, 202, 170),
    "marron": (120, 80, 50),
    "rojo": (200, 30, 40),
    "naranja": (235, 130, 40),
    "amarillo": (240, 210, 60),
    "verde": (50, 140, 70),
    "azul": (40, 80, 180),
    "morado": (120, 50, 150),
    "rosa": (230, 130, 170),
}
