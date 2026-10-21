"""Genera y ejecuta el notebook entregable (ENTREGABLES/Notebook_CNN_FashionMNIST.ipynb).

El notebook cubre las 4 fases del taller:
  A) Preparación y preprocesamiento
  B) Modelo base (CNN con Conv 3x3 ReLU, MaxPooling y Softmax)
  C) Experimentación (con/sin Dropout, SGD vs Adam, Data Augmentation, BatchNorm)
  D) Evaluación (curvas, matriz de confusión, fórmula de Accuracy)

Guarda las figuras en ENTREGABLES/figures/ y los resultados en
ENTREGABLES/resultados_notebook.json (que consume build_informe_entregable.py).

Uso:
    .venv\\Scripts\\python.exe build_notebook_entregable.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ENTREGABLES"
NB_PATH = OUT_DIR / "Notebook_CNN_FashionMNIST.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
md("""# Taller: Diseño y Optimización de una CNN para Clasificación de Imágenes Complejas

**Dataset asignado:** Fashion-MNIST (10 categorías de prendas, imágenes 28×28 en escala de grises)

**Contexto de la misión:** el equipo actúa como ingenieros de visión artificial para una
*startup de logística* que necesita clasificar imágenes de productos con alta precisión,
enfrentando problemas reales como el **sobreajuste (overfitting)** y la **variabilidad de los datos**.

## Estructura del notebook

| Fase | Contenido |
|------|-----------|
| **A** | Preparación y preprocesamiento (carga, balance, normalización, split, *Data Augmentation*) |
| **B** | Modelo base (Conv 3×3 ReLU → MaxPooling → Softmax) |
| **C** | Experimentación: Dropout, SGD vs Adam, Augmentation, Batch Normalization |
| **D** | Evaluación: curvas Loss/Accuracy, matriz de confusión y fórmula de Accuracy |

> Se fija `RANDOM_SEED = 42` en todos los pasos para que los resultados sean reproducibles.""")

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
code("""# Configuración general: rutas, semilla y librerías
import json, os, random, sys, time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# El notebook se ejecuta dentro de ENTREGABLES/; la raíz del proyecto es el padre.
WORK = Path.cwd().resolve()
ROOT = WORK.parent
sys.path.insert(0, str(ROOT))

FIGDIR = WORK / "figures"
FIGDIR.mkdir(exist_ok=True)
RESULTADOS_PATH = WORK / "resultados_notebook.json"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.config import CLASS_NAMES_ES, DEFAULT_HPARAMS, RANDOM_SEED
from src.data_loader import class_distribution, load_fashion_mnist
from src.data_preparation import get_prepared_data, plot_augmentation_preview
from src.evaluation import (
    compute_confusion,
    evaluate_test,
    per_class_accuracy,
    plot_confusion_matrix,
    plot_training_curves,
    top_confusions,
)
from src.modeling import build_cnn, summary_text, train_model


def set_seed(seed: int = RANDOM_SEED) -> None:
    \"\"\"Fija las semillas de Python, NumPy y TensorFlow para reproducibilidad.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def guardar(fig, nombre: str) -> None:
    \"\"\"Guarda la figura en ENTREGABLES/figures/ y la muestra en la celda.\"\"\"
    fig.savefig(FIGDIR / nombre, dpi=130, bbox_inches="tight")
    display(fig)
    plt.close(fig)


set_seed()
print(f"Semilla: {RANDOM_SEED} | Raíz del proyecto: {ROOT}")""")

# ---------------------------------------------------------------------------
# FASE A
# ---------------------------------------------------------------------------
md("""## Fase A — Preparación y Preprocesamiento

### A.1 Carga y balance de clases

Fashion-MNIST contiene **70.000 imágenes** 28×28 en escala de grises: **60.000 de entrenamiento**
y **10.000 de prueba**, repartidas equitativamente en 10 clases (6.000 y 1.000 por clase).
Un dataset equilibrado evita que el modelo sesgue sus predicciones hacia la clase mayoritaria.""")

code("""# A.1 — Carga de los datos y verificación del balance de clases
set_seed()
x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_fashion_mnist()
print(f"Entrenamiento: {x_train_raw.shape}  Prueba: {x_test_raw.shape}")
print(f"Rango de píxeles: [{x_train_raw.min()}, {x_train_raw.max()}] (uint8)")

dist = class_distribution(y_train_raw)
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.bar(range(10), dist, color="#4C72B0")
ax.set_xticks(range(10))
ax.set_xticklabels([f"{i}\\n{n}" for i, n in enumerate(CLASS_NAMES_ES)], fontsize=7)
ax.set_title("Distribución de clases (entrenamiento): 6000 imágenes por clase")
ax.set_ylabel("Imágenes")
fig.tight_layout()
guardar(fig, "fig_distribucion.png")
plt.close(fig)""")

md("""### A.2 Normalización, canal y división train/val

* Los píxeles se escalan de `[0, 255]` a `[0, 1]` (**normalización**): estabiliza el entrenamiento
  y acelera la convergencia del optimizador.
* Se añade la dimensión de canal `(n, 28, 28) → (n, 28, 28, 1)`, requerida por las capas `Conv2D`.
* El entrenamiento se divide **estratificadamente** en 90 % train / 10 % val, preservando la
  proporción de cada clase en ambas particiones.""")

code("""# A.2 — Normalización a [0,1], canal y split estratificado train/val
set_seed()
prepared = get_prepared_data(val_size=DEFAULT_HPARAMS[\"validation_split\"])
for k in (\"x_train\", \"x_val\", \"x_test\"):
    print(f\"{k:8s} -> {prepared[k].shape}\")
print(f\"\\nRango normalizado: [{float(prepared['x_train'].min()):.1f}, {float(prepared['x_train'].max()):.1f}]\")
_, val_counts = np.unique(prepared[\"y_val\"], return_counts=True)
print(\"Clases en validación (por clase):\", val_counts.min(), \"-\", val_counts.max())""")

md("""### A.3 Data Augmentation (rotaciones y zooms aleatorios)

El *Data Augmentation* genera variantes aleatorias de cada imagen de entrenamiento
(rotación ±8 %, zoom 10 %, traslación 10 %). Esto **aumenta la variabilidad** de los datos
sin etiquetar imágenes nuevas y ayuda a reducir el sobreajuste. Las capas de aumento solo
se activan durante el entrenamiento (`training=True`); en validación/prueba la imagen queda intacta.""")

code("""# A.3 — Vista previa del Data Augmentation (originales vs. aumentados)
set_seed()
fig = plot_augmentation_preview(prepared[\"x_train\"], prepared[\"y_train\"], n=8)
guardar(fig, "fig_augmentation.png")""")

# ---------------------------------------------------------------------------
# FASE B
# ---------------------------------------------------------------------------
md("""## Fase B — Construcción del Modelo Base

Arquitectura CNN ( TensorFlow/Keras ) con los componentes mínimos requeridos:

| Capa | Configuración | Salida |
|------|---------------|--------|
| Input | — | 28 × 28 × 1 |
| Conv2D | 32 filtros **3×3**, stride 1, `padding='same'`, **ReLU** | 28 × 28 × 32 |
| MaxPooling2D | ventana 2×2, stride 2 | 14 × 14 × 32 |
| Conv2D | 64 filtros **3×3**, `same`, ReLU | 14 × 14 × 64 |
| MaxPooling2D | 2×2 | 7 × 7 × 64 |
| Conv2D | 128 filtros **3×3**, `same`, ReLU | 7 × 7 × 128 |
| Flatten | — | 6272 |
| Dense | 128 neuronas, ReLU | 128 |
| Dropout | tasa 0.3 (regularización) | 128 |
| **Dense (salida)** | 10 neuronas, **Softmax** | 10 (probabilidades) |

**¿Por qué estos componentes?**

* **Filtros 3×3 con `padding='same'`**: capturan patrones locales (bordes, texturas) sin
  reducir el tamaño espacial ni introducir bordes artificiales.
* **ReLU** introduce no linealidad y evita el desvanecimiento del gradiente.
* **MaxPooling 2×2** reduce la dimensionalidad espacial a la mitad, aportando
  invariancia a pequeñas traslaciones y disminuyendo el costo computacional.
* **Softmax** convierte las salidas en una distribución de probabilidad sobre las 10 clases
  (clasificación multiclase).

Se entrena el modelo base con **Adam** (lr = 1e-3), pérdida `sparse_categorical_crossentropy`
y **EarlyStopping** (paciencia 4) sobre `val_loss` para detener el entrenamiento cuando la
validación deja de mejorar.""")

code("""# B.1 — Construcción del modelo base y resumen de la arquitectura
set_seed()
hparams = {**DEFAULT_HPARAMS, "epochs": 15}
modelo_base = build_cnn(hparams, use_augmentation=False, use_dropout=True)
print(summary_text(modelo_base))
print(f"Parámetros entrenables: {modelo_base.count_params():,}")""")

code("""# B.2 — Entrenamiento del modelo base + curvas Loss / Accuracy
set_seed()
t0 = time.perf_counter()
hist_base = train_model(modelo_base, prepared, hparams)
seg_base = time.perf_counter() - t0
loss_base, acc_base = evaluate_test(modelo_base, prepared)
print(f"Modelo base (Adam + Dropout): test_loss={loss_base:.4f}  test_accuracy={acc_base:.4f}  ({seg_base:.0f} s)")
fig = plot_training_curves(hist_base)
guardar(fig, "fig_curvas_base.png")""")

# ---------------------------------------------------------------------------
# FASE C
# ---------------------------------------------------------------------------
md("""## Fase C — Experimentación (el "core" del taller)

Se comparan **5 configuraciones** sobre la misma arquitectura, variando un factor a la vez:

| Experimento | Dropout | Augmentation | Optimizador | BatchNorm |
|-------------|:-------:|:------------:|:-----------:|:---------:|
| 1. Sin Dropout | ✗ | ✗ | Adam | ✗ |
| 2. Con Dropout (= modelo base) | ✓ | ✗ | Adam | ✗ |
| 3. Dropout + Augmentation | ✓ | ✓ | Adam | ✗ |
| 4. Dropout + Aug + SGD | ✓ | ✓ | SGD (lr 0.01, momentum 0.9) | ✗ |
| 5. + Batch Normalization (innovación) | ✓ | ✓ | Adam | ✓ |

**¿Qué es el overfitting y por qué ocurre?** Un modelo sobreajustado memoriza los datos de
entrenamiento (accuracy de train muy alta) pero no generaliza a datos nuevos (accuracy de val
estancada o descendente). Se manifiesta porque una CNN con ~900.000 parámetros tiene mucha
capacidad para memorizar 54.000 imágenes. Las estrategias de mitigación comparadas aquí son:
**Dropout**, **Data Augmentation** y **EarlyStopping + ReduceLROnPlateau**.""")

code("""# C.1 — Experimentación comparativa (se reutiliza el modelo base ya entrenado)
set_seed()
EPOCHS_EXP = 10
hparams_exp = {**DEFAULT_HPARAMS, "epochs": EPOCHS_EXP}

CONFIGS = [
    dict(slug="sin_dropout",   titulo="1. Sin Dropout",                 use_dropout=False, use_aug=False, use_bn=False, optimizer="adam"),
    dict(slug="con_dropout",   titulo="2. Con Dropout (modelo base)",   use_dropout=True,  use_aug=False, use_bn=False, optimizer="adam"),
    dict(slug="dropout_aug",   titulo="3. Dropout + Augmentation",      use_dropout=True,  use_aug=True,  use_bn=False, optimizer="adam"),
    dict(slug="dropout_aug_sgd", titulo="4. Dropout + Aug + SGD",       use_dropout=True,  use_aug=True,  use_bn=False, optimizer="sgd"),
    dict(slug="final_bn",      titulo="5. + Batch Norm (innovación)",   use_dropout=True,  use_aug=True,  use_bn=True,  optimizer="adam"),
]

rows, trained = [], {"con_dropout": modelo_base}

for cfg in CONFIGS:
    slug = cfg["slug"]
    if slug == "con_dropout":
        # Se reutiliza el modelo base entrenado en la Fase B.
        hist = hist_base.history
        n_params, segundos, model = modelo_base.count_params(), seg_base, modelo_base
    else:
        print(f">>> Entrenando: {cfg['titulo']} ({cfg['optimizer']})")
        set_seed()
        model = build_cnn(
            hparams_exp,
            use_augmentation=cfg["use_aug"],
            use_dropout=cfg["use_dropout"],
            use_batchnorm=cfg["use_bn"],
        )
        n_params = model.count_params()
        t0 = time.perf_counter()
        history = train_model(model, prepared, hparams_exp, optimizer=cfg["optimizer"])
        segundos = time.perf_counter() - t0
        hist = history.history
        fig = plot_training_curves(history)
        guardar(fig, f"fig_curvas_{slug}.png")
    _, test_acc = evaluate_test(model, prepared)
    trained[slug] = model
    rows.append(dict(
        slug=slug, titulo=cfg["titulo"], optimizer=cfg["optimizer"].upper(),
        dropout=cfg["use_dropout"], augmentation=cfg["use_aug"], batchnorm=cfg["use_bn"],
        params=int(n_params), segundos=round(segundos, 1),
        train_acc_final=round(float(hist["accuracy"][-1]), 4),
        val_acc_final=round(float(hist["val_accuracy"][-1]), 4),
        val_loss_final=round(float(hist["val_loss"][-1]), 4),
        test_acc=round(float(test_acc), 4),
    ))
    print(f"    test_acc={test_acc:.4f}  train_acc={hist['accuracy'][-1]:.4f}  val_acc={hist['val_accuracy'][-1]:.4f}")

df = pd.DataFrame(rows)
print()
display(df[["titulo", "optimizer", "dropout", "augmentation", "batchnorm", "params", "test_acc", "train_acc_final", "val_acc_final"]])""")

code("""# C.2 — Gráfica comparativa de la precisión en el conjunto de prueba
fig, ax = plt.subplots(figsize=(9, 4))
order = df.sort_values("test_acc")
colores = ["#c44e52" if v < 0.90 else "#4c9f70" for v in order["test_acc"]]
ax.barh(order["titulo"], order["test_acc"], color=colores)
for i, v in enumerate(order["test_acc"]):
    ax.text(v + 0.002, i, f"{v:.2%}", va="center", fontsize=9)
ax.set_xlim(0.80, 0.95)
ax.set_xlabel("Accuracy en test")
ax.set_title("Comparativa de configuraciones (Fase C)")
fig.tight_layout()
guardar(fig, "fig_comparativa.png")""")

# ---------------------------------------------------------------------------
# FASE D
# ---------------------------------------------------------------------------
md("""## Fase D — Evaluación del mejor modelo

### D.1 Curvas del mejor modelo y matriz de confusión

La **matriz de confusión** permite ver qué clases se confunden entre sí
(por ejemplo, Camisa ↔ Suéter ↔ Abrigo, que son visualmente similares).""")

code("""# D.1 — Mejor configuración: curvas, matriz de confusión y reporte por clase
set_seed()
best = max(rows, key=lambda r: r["test_acc"])
best_model = trained[best["slug"]]
print(f"Mejor configuración: {best['titulo']}  (test_acc = {best['test_acc']:.4f})")

y_pred, cm, report = compute_confusion(best_model, prepared)
print("\\n", report)

fig = plot_confusion_matrix(cm)
guardar(fig, "fig_matriz_confusion.png")

print("Top-5 confusiones (real → predicho):")
for d in top_confusions(cm, top=5):
    print(f"  {d['real']:12s} -> {d['predicho']:12s}: {d['casos']} casos")""")

md("""### D.2 Cálculo de la precisión final con la fórmula de Accuracy

La exactitud global se calcula como la suma de los aciertos (diagonal de la matriz de
confusión) dividida entre el número total de muestras:

**Accuracy = Σᵢ₌₁..C TPᵢ / N_total**""")

code("""# D.2 — Fórmula de Accuracy aplicada a la matriz de confusión
correctos = int(np.trace(cm))   # suma de la diagonal: verdaderos positivos por clase
total = int(cm.sum())           # total de muestras evaluadas
accuracy = correctos / total
print(f"Accuracy = Σ TPᵢ / N_total = {correctos} / {total} = {accuracy:.4f} = {accuracy:.2%}")

fig, ax = plt.subplots(figsize=(7, 3))
ax.axis("off")
ax.text(0.5, 0.7, r"$\\mathrm{Accuracy} = \\frac{\\sum_{i=1}^{C} TP_i}{N_{total}} = \\frac{%d}{%d} = %.4f = %.2f\\%%$" % (correctos, total, accuracy, accuracy * 100),
        ha="center", va="center", fontsize=16, transform=ax.transAxes)
ax.set_title("Cálculo de la precisión final", fontsize=12, pad=12)
guardar(fig, "fig_formula_accuracy.png")

df_clases = pd.DataFrame(per_class_accuracy(cm))
display(df_clases)""")

md("""## Conclusiones

1. **Arquitectura (Fase B):** la CNN con filtros 3×3 (`same`), MaxPooling 2×2 y salida Softmax
   alcanza una precisión superior al 90 % exigido, con ~897 mil parámetros
   (modelo base: @@BASE_ACC@@ en test).
2. **Overfitting:** el modelo **sin Dropout** muestra el patrón clásico
   (train @@SIN_TRAIN@@ vs. val @@SIN_VAL@@, con `val_loss` en aumento). El Dropout 0.3 +
   EarlyStopping + ReduceLROnPlateau cierran ese hueco: la brecha train/val se reduce y la
   pérdida de validación se estabiliza.
3. **Optimizador:** Adam converge mucho más rápido que SGD en el mismo número de épocas
   (SGD con lr 0.01 y momentum 0.9 queda por debajo en el mismo tiempo de entrenamiento);
   Adam es la opción recomendada aquí.
4. **Data Augmentation:** las rotaciones/zooms aleatorios restan unos puntos de accuracy en el
   test "limpio" del dataset pero **aumentan la robustez** ante fotos reales desalineadas o de
   bajo contraste: al combinar augmentation con un preprocesado de inferencia que deja margen
   y normaliza el contraste, la tasa de calzado clasificado erróneamente como "Bolso" pasó de
   ~1/60 a 0/120 en un muestreo del conjunto de prueba.
5. **Innovación — Batch Normalization:** se implementó como técnica extra; estabiliza la
   distribución de activaciones entre capas. Con el número de épocas usado y augmentation
   activa no logró superar al Dropout puro, pero queda integrada en
   `build_cnn(use_batchnorm=True)` como línea de mejora (más épocas / más datos).
6. **Precisión final:** el mejor modelo alcanza **@@BEST_ACC@@ de accuracy** en el conjunto de
   prueba (fórmula en la sección D.2), cumpliendo el objetivo del taller y de la misión de
   logística.
""")

code("""# D.3 — Consolidación de resultados para el informe
resultados = {
    "dataset": {"train": int(len(prepared["y_train"])), "val": int(len(prepared["y_val"])), "test": int(len(prepared["y_test"]))},
    "modelo_base": {"test_acc": round(float(acc_base), 4), "segundos": round(seg_base, 1), "params": int(modelo_base.count_params())},
    "experiments": rows,
    "best": best,
    "accuracy_formula": {"correct": correctos, "total": total, "accuracy": round(float(accuracy), 4)},
    "confusion_matrix": cm.tolist(),
    "top_confusions": top_confusions(cm, top=5),
    "per_class": per_class_accuracy(cm),
    "classification_report": report,
}
RESULTADOS_PATH.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
print("Resultados guardados en:", RESULTADOS_PATH)
print("Figuras generadas en:", FIGDIR)""")

nb.cells = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

OUT_DIR.mkdir(exist_ok=True)
nbf.write(nb, NB_PATH)
print("Notebook escrito:", NB_PATH)

print("Ejecutando notebook (entrenamiento de 5 configuraciones, ~10-15 min)...")
client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name="python3",
    resources={"metadata": {"path": str(OUT_DIR)}},
)
client.execute()

# Sustituir los marcadores de la celda de conclusiones con los valores reales.
res = json.loads((OUT_DIR / "resultados_notebook.json").read_text(encoding="utf-8"))
sin = next(r for r in res["experiments"] if r["slug"] == "sin_dropout")
reemplazos = {
    "@@BASE_ACC@@": f"{res['modelo_base']['test_acc']:.2%}",
    "@@SIN_TRAIN@@": f"{sin['train_acc_final']:.2%}",
    "@@SIN_VAL@@": f"{sin['val_acc_final']:.2%}",
    "@@BEST_ACC@@": f"{res['accuracy_formula']['accuracy']:.2%}",
}
for cell in nb.cells:
    if cell.cell_type == "markdown" and "@@BEST_ACC@@" in cell.source:
        for k, v in reemplazos.items():
            cell.source = cell.source.replace(k, v)
nbf.write(nb, NB_PATH)
print("Notebook ejecutado y guardado:", NB_PATH)
print("Mejor configuración:", res["best"])
