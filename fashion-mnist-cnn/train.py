"""Script CLI: entrena la CNN (y el baseline MLP) y guarda los modelos.

Uso:
    .venv\\Scripts\\python.exe train.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from src.config import BASELINE_PATH, DEFAULT_HPARAMS, MODEL_PATH, RANDOM_SEED
from src.data_preparation import get_prepared_data
from src.evaluation import evaluate_test
from src.modeling import (
    build_baseline_mlp,
    build_cnn,
    compile_model,
    save_model,
    summary_text,
    train_model,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    import random

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    set_seed()
    print("[1/4] Preparando datos...")
    prepared = get_prepared_data(val_size=DEFAULT_HPARAMS["validation_split"])
    print(f"      train={prepared['x_train'].shape} val={prepared['x_val'].shape} test={prepared['x_test'].shape}")

    print("[2/4] Construyendo CNN...")
    model = build_cnn(DEFAULT_HPARAMS)
    print(summary_text(model))

    print("[3/4] Entrenando CNN...")
    start = time.perf_counter()
    history = train_model(model, prepared, DEFAULT_HPARAMS)
    elapsed = time.perf_counter() - start
    save_model(model, MODEL_PATH)
    loss, acc = evaluate_test(model, prepared)
    print(f"      CNN -> test_loss={loss:.4f} test_accuracy={acc:.4f} ({elapsed:.1f}s)")

    print("[4/4] Entrenando baseline MLP...")
    baseline = build_baseline_mlp()
    baseline_hparams = {**DEFAULT_HPARAMS, "epochs": 8}
    baseline_history = train_model(baseline, prepared, baseline_hparams)
    save_model(baseline, BASELINE_PATH)
    b_loss, b_acc = evaluate_test(baseline, prepared)
    print(f"      MLP -> test_loss={b_loss:.4f} test_accuracy={b_acc:.4f}")

    np.save(MODEL_PATH.with_suffix(".history.npy"), history.history, allow_pickle=True)
    print(f"\nModelos guardados en:\n  {MODEL_PATH}\n  {BASELINE_PATH}")


if __name__ == "__main__":
    main()
