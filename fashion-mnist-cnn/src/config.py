"""Configuración central del proyecto Fashion-MNIST (CRISP-ML + Scrum)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "archive"
TRAIN_CSV: Path = DATA_DIR / "fashion-mnist_train.csv"
TEST_CSV: Path = DATA_DIR / "fashion-mnist_test.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "fashion_cnn.keras"
BASELINE_PATH: Path = MODELS_DIR / "baseline_mlp.keras"

IMG_SIZE: int = 28
IMG_SHAPE: tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, 1)
NUM_CLASSES: int = 10

CLASS_NAMES_ES: list[str] = [
    "Camiseta/top",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
]
CLASS_NAMES_EN: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Hiperparámetros por defecto del modelo CNN
DEFAULT_HPARAMS: dict = {
    "conv1_filters": 32,
    "conv2_filters": 64,
    "kernel_size": 3,
    "dense_units": 128,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 20,
    "validation_split": 0.1,
}

# Objetivos de negocio / KPIs (Fase 1 - Comprensión del Negocio)
BUSINESS_KPIS: list[dict] = [
    {"kpi": "Accuracy en test", "objetivo": "≥ 90%", "baseline": "MLP ~88%"},
    {"kpi": "Latencia de inferencia", "objetivo": "< 50 ms/imagen", "baseline": "N/A"},
    {"kpi": "Tamaño del modelo", "objetivo": "< 5 MB", "baseline": "N/A"},
    {"kpi": "F1-macro", "objetivo": "≥ 0.90", "baseline": "MLP ~0.88"},
]

RANDOM_SEED: int = 42
