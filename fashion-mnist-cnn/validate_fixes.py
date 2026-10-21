"""Validación de las mejoras de preprocesado y modelo de producción.

Compara los resultados ANTES (modelo antiguo + preprocesado antiguo) y DESPUÉS
(modelo de producción + preprocesado con margen y contraste) en:
  1. Accuracy en el test set de Fashion-MNIST.
  2. Tasa de calzado (Sandalia/Zapatilla/Botín) clasificado como Bolso con el
     pipeline de subida de la GUI.
  3. Acierto top-1/top-3 sobre las fotos reales de `fallos/`.
  4. Latencia media de inferencia.

Uso:
    .venv\\Scripts\\python.exe validate_fixes.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES_ES, MODEL_PATH
from src.data_loader import load_fashion_mnist
from src.deployment import predict, preprocess_array, preprocess_uploaded
from src.evaluation import evaluate_test
from src.modeling import load_model

# Etiqueta real de cada foto de fallos/ (definida por el usuario).
FALLOS_GT = {
    "images (1)": "Zapatilla",       # tenis deportivos de futbol
    "images (2)": "Pantalón",        # jeans
    "images (3)": "Abrigo",          # casaca de viento
    "images (4)": "Sandalia",
    "images (5)": "Camisa",
    "png-transparent-jeans": "Pantalón",     # jeans roto/desgastado
    "png-transparent-women": "Abrigo",       # pea coat
    "-new_desc": "Botín",            # zapatos de vestir (john holden)
}

# Resultados medidos ANTES de las mejoras (modelo 90.67% + preprocesado previo).
BEFORE = {
    "test_accuracy": 0.9067,
    "footwear_as_bag": "1/60",
    "fallos_top1": "4/8",
    "fallos_fallan": [
        "tenis de futbol -> Bolso 64%",
        "sandalia -> Bolso 91%",
        "abrigo pea-coat -> Vestido 37%",
        "zapatos de vestir -> Sandalia 99%",
    ],
}


def main() -> None:
    model = load_model(MODEL_PATH)
    if model is None:
        raise SystemExit("No hay modelo en models/fashion_cnn.keras. Ejecuta train_final.py.")

    print(f"Modelo de producción: {MODEL_PATH.name}")

    # 1) Accuracy en test.
    x_train, y_train, x_test, y_test = load_fashion_mnist()
    prepared = {
        "x_test": preprocess_array(x_test)[..., np.newaxis],
        "y_test": y_test,
    }
    _, test_acc = evaluate_test(model, prepared)
    print(f"\n1) Accuracy test: {BEFORE['test_accuracy']:.4f} -> {test_acc:.4f}")

    # 2) Calzado clasificado como Bolso con el pipeline de subida.
    rng = np.random.RandomState(0)
    fw_idx = np.where(np.isin(y_test, [5, 7, 9]))[0]
    sample = fw_idx[rng.choice(len(fw_idx), 120, replace=False)]
    preds = []
    for i in sample:
        t = preprocess_uploaded(Image.fromarray(x_test[i]))
        preds.append(int(model.predict(t.reshape(1, 28, 28, 1), verbose=0)[0].argmax()))
    preds = np.array(preds)
    as_bag = int((preds == 8).sum())
    print(f"2) Calzado -> Bolso (pipeline de subida): {BEFORE['footwear_as_bag']} -> {as_bag}/120")

    # 3) Fotos reales de fallos/.
    fallos_dir = MODEL_PATH.parent.parent / "fallos"
    hits_top1 = hits_top3 = 0
    print("\n3) Fotos reales (fallos/):")
    if fallos_dir.exists():
        for f in sorted(fallos_dir.iterdir()):
            gt = next((v for k, v in FALLOS_GT.items() if f.name.startswith(k)), None)
            if gt is None:
                continue
            r = predict(model, preprocess_uploaded(Image.open(f)))
            ok_top1 = r["recognized"] and r["raw_label"] == gt
            ok_top3 = gt in [t["label"] for t in r["top3"]]
            hits_top1 += ok_top1
            hits_top3 += ok_top3
            estado = "OK" if ok_top1 else ("top-3" if ok_top3 else "FALLA")
            print(f"   {f.name[:40]:42s} real={gt:10s} "
                  f"top1={r['raw_label']:10s} {r['confidence']:.0%}  {estado}")
        print(f"   top-1: {BEFORE['fallos_top1']} -> {hits_top1}/8   top-3: {hits_top3}/8")
    else:
        print("   (no existe la carpeta fallos/; se omite)")

    # 4) Latencia media.
    t = preprocess_uploaded(Image.fromarray(x_test[0]))
    lat = [predict(model, t)["latency_ms"] for _ in range(20)]
    print(f"\n4) Latencia media: {np.mean(lat):.1f} ms (objetivo < 50 ms)")

    print("\nAntes, fallaban:")
    for x in BEFORE["fallos_fallan"]:
        print(f"   - {x}")


if __name__ == "__main__":
    main()
