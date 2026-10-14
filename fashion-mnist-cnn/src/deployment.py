"""Fase 6 - Despliegue (inferencia del modelo)."""
from __future__ import annotations

import time

import numpy as np

from .config import CLASS_NAMES_ES, IMG_SIZE


def preprocess_uploaded(pil_image, invert: str = "auto") -> np.ndarray:
    """Convierte una imagen subida a un tensor (28, 28, 1) normalizado.

    Fashion-MNIST usa prendas claras sobre fondo negro; si la imagen tiene
    fondo claro se invierte automáticamente para asemejarse al dataset.
    """
    img = pil_image.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype="float32")
    if invert == "auto":
        if arr.mean() > 127:
            arr = 255.0 - arr
    elif invert:
        arr = 255.0 - arr
    arr = arr / 255.0
    return arr[..., np.newaxis]


def preprocess_array(arr: np.ndarray) -> np.ndarray:
    """Normaliza un array (28, 28) o (28, 28, 1) ya en escala 0-255."""
    arr = arr.astype("float32")
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr


def predict(model, image: np.ndarray) -> dict:
    """Ejecuta la inferencia sobre una imagen (28, 28, 1)."""
    batch = image[np.newaxis, ...] if image.ndim == 3 else image
    start = time.perf_counter()
    proba = model.predict(batch, verbose=0)[0]
    latency_ms = (time.perf_counter() - start) * 1000
    index = int(np.argmax(proba))
    return {
        "index": index,
        "label": CLASS_NAMES_ES[index],
        "confidence": float(proba[index]),
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
