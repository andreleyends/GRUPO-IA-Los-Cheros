"""
Configuración global para el proyecto del Equipo 6 - Familia 3 (Probabilísticos y Estructurados).

Define semillas, rutas, parámetros comunes y constantes reutilizadas
en todos los módulos.
"""

from pathlib import Path
import os

# Semilla global
SEED = 42

# Rutas del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"
APP_DIR = ROOT_DIR / "app"
SRC_DIR = ROOT_DIR / "src"

# Crear directorios si no existen
for _dir in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, DOCS_DIR, APP_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Parámetros de división
TEST_SIZE = 0.2
CV_FOLDS = 5

# Parámetros para visualizaciones
DPI_FIG = 300
FIGSIZE_STD = (10, 6)
FIGSIZE_WIDE = (12, 8)

# Configuración para evitar fugas de datos
RANDOM_STATE = SEED

# Marcador para indicar que no se debe escalar para NB Multinomial
SCALING_NOT_APPLICABLE_NB = True
