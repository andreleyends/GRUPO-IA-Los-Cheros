"""Fase 6 - Despliegue (inferencia del modelo)."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from .config import CLASS_NAMES_ES, IMG_SIZE


def _background_level(arr: np.ndarray) -> float:
    """Estima el nivel de fondo a partir de los píxeles del borde.

    El borde de la imagen casi siempre es fondo, por lo que es un indicador
    mucho más fiable que la media global (que falla si el objeto llena el
    cuadro o el fondo es oscuro).
    """
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.median(border))


def _crop_and_center(arr: np.ndarray, fill: float = 0.8) -> np.ndarray:
    """Recorta el objeto a su bounding box y lo centra en un lienzo cuadrado.

    El objeto ocupa como máximo `fill` del lienzo (deja margen). En
    Fashion-MNIST las prendas no tocan los bordes; si el recorte las estira
    hasta el borde, un zapato o sandalia se convierte en una "mancha" que
    llena el cuadro y el modelo la clasifica como Bolso.
    """
    mask = arr > max(float(arr.max()) * 0.15, 20.0)
    if mask.any():
        ys, xs = np.where(mask)
        arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = arr.shape
    side = max(h, w)
    if fill > 0:
        side = int(np.ceil(side / fill))
    canvas = np.zeros((side, side), dtype="float32")
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = arr
    return canvas


def _normalize_contrast(arr: np.ndarray) -> np.ndarray:
    """Estira el contraste del objeto al rango completo [0, 255].

    Fotos con poco contraste producen siluetas casi invisibles tras bajar a
    28x28 (el modelo las ve "fantasma" y falla). Se estira usando los
    percentiles 2 y 98 de los píxeles del objeto para ignorar ruido.
    """
    hi = float(arr.max())
    if hi <= 0:
        return arr
    fg = arr[arr > hi * 0.2]
    if fg.size < 20:
        return arr
    lo_p, hi_p = np.percentile(fg, (2, 98))
    if hi_p - lo_p < 1.0:
        return arr
    out = (arr - lo_p) * (255.0 / (hi_p - lo_p))
    return np.clip(out, 0.0, 255.0)


def _should_invert(arr: np.ndarray) -> bool:
    """Decide si hay que invertir para dejar prenda clara sobre fondo oscuro.

    Se apoya en el nivel de fondo (borde): se invierte si el fondo es claro,
    o si el objeto es más oscuro que el fondo (media global < fondo), lo que
    cubre fondos grises medios donde un umbral fijo fallaría.
    """
    bg = _background_level(arr)
    return bg > 127 or float(arr.mean()) < bg


def preprocess_uploaded(
    pil_image,
    invert: str = "auto",
    clean_background: bool = True,
    crop: bool = True,
    enhance_contrast: bool = True,
) -> np.ndarray:
    """Convierte una imagen subida a un tensor (28, 28, 1) normalizado.

    Fashion-MNIST usa prendas claras sobre fondo oscuro. Para acercar una foto
    real a esa distribución se aplica:
      1. Escala de grises.
      2. Normalización de polaridad (fondo claro -> se invierte), decidida por
         el nivel de fondo del borde, no por la media global.
      3. Limpieza del fondo (opcional) para eliminar ruido/complejidad.
      4. Recorte al objeto con margen (el objeto no toca los bordes) y
         centrado en un lienzo cuadrado.
      5. Normalización de contraste del objeto (percentiles 2-98).
      6. Redimensionado a 28x28 y normalización a [0, 1].
    """
    arr = np.asarray(pil_image.convert("L"), dtype="float32")

    # 1) Polaridad: buscamos fondo oscuro y prenda clara.
    if invert == "auto":
        do_invert = _should_invert(arr)
    elif invert in ("sí", "si", "yes", "true", "1"):
        do_invert = True
    else:
        do_invert = False
    if do_invert:
        arr = 255.0 - arr

    # 2) Limpieza de fondo: se elimina lo que queda al nivel del fondo.
    if clean_background and arr.size:
        bg_after = _background_level(arr)
        hi = float(arr.max())
        if hi > bg_after:
            threshold = bg_after + (hi - bg_after) * 0.25
            arr = np.where(arr > threshold, arr, 0.0)

    # 3) Recorte y centrado del objeto (con margen).
    if crop:
        arr = _crop_and_center(arr)

    # 4) Contraste del objeto al rango completo.
    if enhance_contrast:
        arr = _normalize_contrast(arr)

    # 5) Redimensionado final y normalización.
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
    )
    out = np.asarray(img, dtype="float32") / 255.0
    return out[..., np.newaxis]


def preprocess_array(arr: np.ndarray) -> np.ndarray:
    """Normaliza un array (28, 28) o (28, 28, 1) ya en escala 0-255."""
    arr = arr.astype("float32")
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr


def predict(model, image: np.ndarray, min_confidence: float = 0.5) -> dict:
    """Ejecuta la inferencia sobre una imagen (28, 28, 1).

    Si la probabilidad máxima es menor que `min_confidence` se marca la
    predicción como "No reconocido" (el softmax siempre asigna una clase, y en
    imágenes fuera de dominio puede equivocarse con mucha confianza).
    Se devuelve también el top-3 de clases.
    """
    batch = image[np.newaxis, ...] if image.ndim == 3 else image
    start = time.perf_counter()
    proba = model.predict(batch, verbose=0)[0]
    latency_ms = (time.perf_counter() - start) * 1000
    order = proba.argsort()[::-1][:3]
    top3 = [
        {
            "index": int(i),
            "label": CLASS_NAMES_ES[int(i)],
            "confidence": float(proba[int(i)]),
        }
        for i in order
    ]
    index = top3[0]["index"]
    confidence = top3[0]["confidence"]
    recognized = confidence >= min_confidence
    return {
        "index": index,
        "label": CLASS_NAMES_ES[index] if recognized else "No reconocido",
        "raw_label": CLASS_NAMES_ES[index],
        "recognized": recognized,
        "confidence": confidence,
        "top3": top3,
        "probabilities": proba,
        "latency_ms": latency_ms,
    }


DEPLOYMENT_NOTES = {
    "local": "streamlit run app.py",
    "cloud": "Streamlit Community Cloud (subir repo + requirements.txt)",
    "contenedor": "Docker + imagen python:3.12-slim + modelo .keras copiado",
    "monitorizacion": [
        "Registrar predicciones y confianza media por sesión.",
        "Detectar data drift si cambian las imágenes de entrada.",
        "Reentrenar con nuevos datos etiquetados de forma periódica (Sprint de mantenimiento).",
        "Re-evaluar KPIs (accuracy ≥ 90%, latencia < 50 ms) tras cada reentreno.",
    ],
}


def load_selected_test_sample(x_test, y_test, index: int):
    return x_test[index][..., np.newaxis], int(y_test[index])
