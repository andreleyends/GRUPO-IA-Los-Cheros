"""Reentrena el modelo final más robusto para despliegue.

Cambios respecto al modelo original:
- Data Augmentation de producción (geometría + volteo horizontal + brillo suave).
- Early stopping y ReduceLROnPlateau.

Tras entrenar compara el nuevo modelo contra el mejor referente
(results/modelo_final.keras, 92.4% de test) y despliega el MEJOR en
models/fashion_cnn.keras (respaldando el anterior en .bak). El historial del
modelo desplegado queda en models/fashion_cnn.history.npy.
"""
from __future__ import annotations

import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random

import numpy as np

from src.config import DEFAULT_HPARAMS, MODEL_PATH, RANDOM_SEED
from src.data_preparation import build_production_augmentation, get_prepared_data
from src.evaluation import evaluate_test
from src.modeling import build_cnn, load_model, save_model, train_model

HISTORY_PATH = MODEL_PATH.with_suffix(".history.npy")
REFERENCE_PATH = MODEL_PATH.parent.parent / "results" / "modelo_final.keras"


def set_seed(seed: int = RANDOM_SEED) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    set_seed()
    print("Preparando datos...")
    prepared = get_prepared_data(val_size=0.1)

    # Precisión del referente actual (modelo de experimentos).
    ref_acc = None
    if REFERENCE_PATH.exists():
        ref_model = load_model(REFERENCE_PATH)
        if ref_model is not None:
            _, ref_acc = evaluate_test(ref_model, prepared)
            print(f"Referente {REFERENCE_PATH.name}: test_acc={ref_acc:.4f}")

    hparams = {**DEFAULT_HPARAMS, "epochs": 30}
    model = build_cnn(
        hparams,
        use_augmentation=True,
        use_dropout=True,
        use_batchnorm=False,
        augmentation=build_production_augmentation(),
    )
    model.summary()

    print(f"\nEntrenando {hparams['epochs']} épocas (early stopping)...")
    history = train_model(model, prepared, hparams, optimizer="adam", early_stopping=True)

    loss, acc = evaluate_test(model, prepared)
    print(f"\nNuevo modelo -> Accuracy en test: {acc:.4f}  |  loss: {loss:.4f}")

    # Elección del modelo de producción: el mejor entre el nuevo y el referente.
    deploy_new = ref_acc is None or acc >= ref_acc
    if deploy_new:
        chosen, chosen_acc, hist = model, acc, history.history
        print(f"Se despliega el NUEVO modelo ({acc:.4f} >= ref {ref_acc if ref_acc is not None else 0:.4f}).")
    else:
        chosen, chosen_acc = load_model(REFERENCE_PATH), ref_acc
        hist = None
        print(f"El referente sigue siendo mejor ({ref_acc:.4f} > {acc:.4f}); se despliega el referente.")

    if MODEL_PATH.exists():
        backup = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".bak")
        shutil.copy2(MODEL_PATH, backup)
        print("Respaldo del modelo anterior:", backup)

    save_model(chosen, MODEL_PATH)
    if hist is not None:
        np.save(HISTORY_PATH, hist)
    print(f"Modelo desplegado: {MODEL_PATH} (test_acc={chosen_acc:.4f})")


if __name__ == "__main__":
    main()
