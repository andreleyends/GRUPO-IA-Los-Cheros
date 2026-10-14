"""Fase 4 - Modelado (CNN)."""
from __future__ import annotations

from pathlib import Path

from .config import BASELINE_PATH, IMG_SHAPE, MODEL_PATH, NUM_CLASSES


def build_cnn(hparams: dict, use_augmentation: bool = True):
    """Construye una red neuronal convolucional para Fashion-MNIST."""
    from tensorflow import keras

    from .data_preparation import build_augmentation

    k = hparams["kernel_size"]
    layers = [keras.layers.Input(shape=IMG_SHAPE)]
    if use_augmentation:
        layers.append(build_augmentation())

    layers += [
        keras.layers.Conv2D(hparams["conv1_filters"], (k, k), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(hparams["conv2_filters"], (k, k), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(hparams["conv2_filters"] * 2, (k, k), activation="relu", padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(hparams["dense_units"], activation="relu"),
        keras.layers.Dropout(hparams["dropout"]),
        keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
    return keras.Sequential(layers, name="cnn_fashion_mnist")


def build_baseline_mlp():
    """Modelo baseline (MLP) para comparar contra la CNN."""
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.Input(shape=IMG_SHAPE),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="baseline_mlp",
    )


def compile_model(model, hparams: dict):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hparams["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def summary_text(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def train_model(model, prepared: dict, hparams: dict, progress_callback=None):
    """Entrena el modelo y devuelve el objeto History."""
    from tensorflow import keras

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    if progress_callback is not None:
        callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: progress_callback(epoch + 1, logs)
            )
        )
    compile_model(model, hparams)
    history = model.fit(
        prepared["x_train"],
        prepared["y_train"],
        validation_data=(prepared["x_val"], prepared["y_val"]),
        epochs=hparams["epochs"],
        batch_size=hparams["batch_size"],
        verbose=0,
        callbacks=callbacks,
    )
    return history


def save_model(model, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_model(path: Path = MODEL_PATH):
    from tensorflow import keras

    if not path.exists():
        return None
    return keras.models.load_model(path)


def model_exists(path: Path = MODEL_PATH) -> bool:
    return path.exists()
