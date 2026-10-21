"""Fase 4 - Modelado (CNN)."""
from __future__ import annotations

from pathlib import Path

from .config import BASELINE_PATH, IMG_SHAPE, MODEL_PATH, NUM_CLASSES


def build_cnn(
    hparams: dict,
    use_augmentation: bool = True,
    use_dropout: bool = True,
    use_batchnorm: bool = False,
    augmentation=None,
):
    """Construye una red neuronal convolucional para Fashion-MNIST.

    Opciones pensadas para la Fase C de experimentación:
    - use_augmentation: activa rotaciones/zoom/traslación (data augmentation).
    - use_dropout: añade la capa Dropout de regularización.
    - use_batchnorm: añade Batch Normalization (técnica extra / innovación).
    - augmentation: capa de aumento personalizada (por defecto build_augmentation).
    """
    from tensorflow import keras

    from .data_preparation import build_augmentation

    k = hparams["kernel_size"]
    layers = [keras.layers.Input(shape=IMG_SHAPE)]
    if use_augmentation:
        layers.append(augmentation if augmentation is not None else build_augmentation())

    layers += [keras.layers.Conv2D(hparams["conv1_filters"], (k, k), strides=(1, 1), padding="same", activation="relu")]
    if use_batchnorm:
        layers.append(keras.layers.BatchNormalization())
    layers.append(keras.layers.MaxPooling2D((2, 2)))

    layers.append(keras.layers.Conv2D(hparams["conv2_filters"], (k, k), strides=(1, 1), padding="same", activation="relu"))
    if use_batchnorm:
        layers.append(keras.layers.BatchNormalization())
    layers.append(keras.layers.MaxPooling2D((2, 2)))

    layers.append(keras.layers.Conv2D(hparams["conv2_filters"] * 2, (k, k), strides=(1, 1), padding="same", activation="relu"))

    layers.append(keras.layers.Flatten())
    layers.append(keras.layers.Dense(hparams["dense_units"], activation="relu"))
    if use_dropout:
        layers.append(keras.layers.Dropout(hparams["dropout"]))
    layers.append(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
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


def compile_model(model, hparams: dict, optimizer: str = "adam"):
    from tensorflow import keras

    if optimizer == "sgd":
        opt = keras.optimizers.SGD(
            learning_rate=hparams.get("sgd_learning_rate", 0.01), momentum=0.9
        )
    else:
        opt = keras.optimizers.Adam(learning_rate=hparams["learning_rate"])

    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def summary_text(model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def train_model(
    model,
    prepared: dict,
    hparams: dict,
    progress_callback=None,
    optimizer: str = "adam",
    early_stopping: bool = True,
):
    """Entrena el modelo y devuelve el objeto History."""
    from tensorflow import keras

    callbacks = []
    if early_stopping:
        callbacks += [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ]
    if progress_callback is not None:
        callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: progress_callback(epoch + 1, logs)
            )
        )
    compile_model(model, hparams, optimizer=optimizer)
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
