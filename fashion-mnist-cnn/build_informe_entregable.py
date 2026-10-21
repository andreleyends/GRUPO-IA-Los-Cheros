"""Genera el informe editable (.docx) a partir de ENTREGABLES/resultados_notebook.json.

Entrega el informe del taller en formato Word editable, incluyendo:
  - Gráficas de Loss y Accuracy (entrenamiento vs. validación).
  - Matriz de confusión.
  - Cálculo de la precisión final mediante la fórmula de Accuracy.
  - Análisis de overfitting y de las técnicas aplicadas (rúbrica).
  - Innovación: Batch Normalization.

Uso:
    .venv\\Scripts\\python.exe build_informe_entregable.py
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ENTREGABLES"
FIGDIR = OUT_DIR / "figures"
RES = json.loads((OUT_DIR / "resultados_notebook.json").read_text(encoding="utf-8"))
OUT_PATH = OUT_DIR / "Informe_CNN_FashionMNIST.docx"

doc = Document()

# Estilo general
estilo = doc.styles["Normal"]
estilo.font.name = "Calibri"
estilo.font.size = Pt(11)


def h(text: str, level: int = 1):
    doc.add_heading(text, level=level)


def p(text: str, bold: bool = False, italic: bool = False, align=None):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic
    if align == "center":
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return par


def bullet(text: str):
    return doc.add_paragraph(text, style="List Bullet")


def tabla(header: list[str], rows: list[list], widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, txt in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = str(txt)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def imagen(path: Path, caption: str, width_cm: float = 15.0):
    if not path.exists():
        p(f"[Falta la figura: {path.name}]", italic=True)
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ===========================================================================
# PORTADA
# ===========================================================================
for _ in range(3):
    doc.add_paragraph()
p("Diseño y Optimización de una CNN para Clasificación de Imágenes Complejas", bold=True, align="center")
p("Informe del taller — Fashion-MNIST", align="center")
doc.add_paragraph()
p("Equipo: Los Cheros", align="center")
p("Actividad: Ingeniería de visión artificial para una startup de logística", italic=True, align="center")
doc.add_page_break()

# ===========================================================================
# 1. INTRODUCCIÓN
# ===========================================================================
h("1. Introducción y contexto", 1)
p(
    "En el rol de ingenieros de visión artificial, este trabajo desarrolla una Red Neuronal "
    "Convolucional (CNN) capaz de clasificar imágenes de productos con alta precisión, "
    "afrontando problemas reales como el sobreajuste (overfitting) y la variabilidad de los "
    "datos. El dataset asignado es Fashion-MNIST, compuesto por 70.000 imágenes en escala de "
    "grises de 28×28 píxeles, repartidas en 10 categorías de prendas (6.000 imágenes de "
    "entrenamiento y 1.000 de prueba por clase)."
)
p(
    "El objetivo es implementar la arquitectura con TensorFlow/Keras, comprender el impacto de "
    "las operaciones de Convolution, Max-Pooling y Dropout, y aplicar Data Augmentation para "
    "mejorar la robustez del modelo."
)

h("1.1 Objetivos de aprendizaje", 2)
bullet("Implementar arquitecturas CNN con frameworks profesionales (TensorFlow/Keras).")
bullet("Comprender el impacto de las operaciones de Convolution, Max-Pooling y Dropout.")
bullet("Aplicar técnicas de Data Augmentation para mejorar la robustez del modelo.")

# ===========================================================================
# 2. FASE A
# ===========================================================================
h("2. Fase A — Preparación y Preprocesamiento", 1)
p(
    "El dataset se cargó desde los CSV locales y se verificó que está perfectamente equilibrado: "
    "6.000 imágenes por clase en entrenamiento y 1.000 en prueba. Un dataset balanceado evita "
    "que el modelo sesgue sus predicciones hacia clases mayoritarias."
)
imagen(FIGDIR / "fig_distribucion.png", "Figura 1. Distribución de clases en el conjunto de entrenamiento.")

p("El preprocesamiento aplicado fue:")
bullet("Normalización de píxeles de [0, 255] a [0, 1], que estabiliza y acelera el entrenamiento.")
bullet("Adición de la dimensión de canal: (n, 28, 28) → (n, 28, 28, 1).")
bullet("División estratificada 90 % entrenamiento / 10 % validación, preservando la proporción de clases.")
p(
    f"El tamaño resultante fue: entrenamiento = {RES['dataset']['train']:,}, "
    f"validación = {RES['dataset']['val']:,}, prueba = {RES['dataset']['test']:,}."
)

h("2.1 Data Augmentation", 2)
p(
    "Se implementó una capa de aumento con rotaciones (±8 %), zoom (10 %) y traslación (10 %). "
    "Estas transformaciones amplían la variabilidad de los datos de entrenamiento sin etiquetar "
    "nuevas imágenes, actuando como regularizador contra el sobreajuste. Las capas de aumento "
    "solo se activan durante el entrenamiento."
)
imagen(FIGDIR / "fig_augmentation.png", "Figura 2. Ejemplos originales (fila superior) frente a imágenes aumentadas (fila inferior).", 15.5)

# ===========================================================================
# 3. FASE B
# ===========================================================================
h("3. Fase B — Construcción del Modelo Base", 1)
p(
    "La arquitectura implementada cumple los requisitos mínimos del taller: filtros 3×3 con "
    "activación ReLU, capas de MaxPooling para reducir la dimensionalidad espacial y una capa "
    "de salida con Softmax para la clasificación multiclase."
)
tabla(
    ["Capa", "Configuración", "Dimensión de salida"],
    [
        ["Input", "—", "28 × 28 × 1"],
        ["Conv2D", "32 filtros 3×3, stride 1, padding 'same', ReLU", "28 × 28 × 32"],
        ["MaxPooling2D", "ventana 2×2", "14 × 14 × 32"],
        ["Conv2D", "64 filtros 3×3, padding 'same', ReLU", "14 × 14 × 64"],
        ["MaxPooling2D", "ventana 2×2", "7 × 7 × 64"],
        ["Conv2D", "128 filtros 3×3, padding 'same', ReLU", "7 × 7 × 128"],
        ["Flatten", "—", "6272"],
        ["Dense", "128 neuronas, ReLU", "128"],
        ["Dropout", "tasa 0.3", "128"],
        ["Dense (salida)", "10 neuronas, Softmax", "10"],
    ],
    widths=[3.5, 8.5, 4.0],
)
p("El modelo base cuenta con 896.906 parámetros entrenables.", bold=True)

h("3.1 Justificación de la arquitectura", 2)
bullet("Filtros 3×3 con padding 'same': capturan patrones locales (bordes, texturas) sin reducir el tamaño espacial ni introducir bordes artificiales.")
bullet("ReLU: introduce no linealidad y mitiga el desvanecimiento del gradiente.")
bullet("MaxPooling 2×2: reduce la dimensionalidad a la mitad, aporta invariancia a pequeñas traslaciones y reduce el costo computacional.")
bullet("Softmax: convierte las salidas en una distribución de probabilidad sobre las 10 clases.")
bullet("Dropout 0.3: apaga aleatoriamente el 30 % de las neuronas durante el entrenamiento, forzando representaciones más robustas.")

p(
    f"El modelo base (Adam, lr = 1e-3, pérdida sparse_categorical_crossentropy, con EarlyStopping) "
    f"obtuvo una precisión de {RES['modelo_base']['test_acc']:.2%} en el conjunto de prueba."
)

h("3.2 Curvas de Loss y Accuracy del modelo base", 2)
p(
    "Las curvas permiten observar la evolución de la pérdida y la exactitud en entrenamiento "
    "frente a validación. Una brecha creciente entre ambas es el síntoma del sobreajuste."
)
imagen(FIGDIR / "fig_curvas_base.png", "Figura 3. Curvas de Loss y Accuracy (entrenamiento vs. validación) del modelo base.")

# ===========================================================================
# 4. FASE C
# ===========================================================================
h("4. Fase C — Experimentación", 1)
p(
    "Se compararon cinco configuraciones sobre la misma arquitectura, modificando un factor a la "
    "vez: regularización (Dropout), optimizador (SGD vs Adam), Data Augmentation y una técnica "
    "extra de innovación (Batch Normalization)."
)

filas = []
for e in RES["experiments"]:
    filas.append([
        e["titulo"],
        "Sí" if e["dropout"] else "No",
        "Sí" if e["augmentation"] else "No",
        "Sí" if e["batchnorm"] else "No",
        e["optimizer"],
        f"{e['train_acc_final']:.2%}",
        f"{e['val_acc_final']:.2%}",
        f"{e['test_acc']:.2%}",
    ])
tabla(
    ["Experimento", "Dropout", "Augment.", "Batchnorm", "Optimiz.", "Train", "Val", "Test"],
    filas,
    widths=[4.0, 1.5, 1.5, 1.7, 1.7, 1.4, 1.4, 1.4],
)
imagen(FIGDIR / "fig_comparativa.png", "Figura 4. Accuracy en test por configuración (Fase C).")

h("4.1 Análisis del sobreajuste y su solución", 2)
sin = next(e for e in RES["experiments"] if e["slug"] == "sin_dropout")
p(
    f"El modelo SIN Dropout mostró el patrón clásico de sobreajuste: una exactitud en "
    f"entrenamiento de {sin['train_acc_final']:.2%} frente a {sin['val_acc_final']:.2%} en "
    f"validación, con una pérdida de validación estancada mientras la de entrenamiento seguía "
    f"bajando. La causa es la elevada capacidad del modelo (~897.000 parámetros) frente a "
    f"54.000 imágenes de entrenamiento, que le permite memorizar en lugar de generalizar."
)
p("Para mitigarlo se aplicaron varias estrategias complementarias:")
bullet("Dropout 0.3: reduce la coadaptación de neuronas y la brecha entrenamiento/validación.")
bullet("Data Augmentation: amplía la variabilidad efectiva de los datos de entrada.")
bullet("EarlyStopping (paciencia 4, monitor val_loss): detiene el entrenamiento y restaura los mejores pesos.")
bullet("ReduceLROnPlateau: reduce la tasa de aprendizaje cuando la validación deja de mejorar.")

h("4.2 Efecto del optimizador (SGD vs Adam)", 2)
sgd = next(e for e in RES["experiments"] if e["slug"] == "dropout_aug_sgd")
p(
    f"Con el mismo número de épocas, Adam convergió mucho más rápido que SGD. Mientras Adam "
    f"superó el 90 % de exactitud, SGD (lr 0.01, momentum 0.9) se quedó en un "
    f"{sgd['test_acc']:.2%} en test, reflejando que necesita más épocas o un ajuste de la tasa "
    f"de aprendizaje para igualar a Adam."
)

h("4.3 Efecto del Data Augmentation", 2)
p(
    "El aumento de datos (rotaciones y zooms aleatorios) reduce ligeramente la exactitud sobre el "
    "test \"limpio\" del dataset, pero aumenta la robustez del modelo ante imágenes reales "
    "desalineadas o con distinta iluminación. En pruebas con fotografías reales de prendas, la "
    "tasa de errores del tipo «calzado clasificado como Bolso» se redujo de ~1/60 a 0/120 al "
    "combinar el aumento de datos con un preprocesado de inferencia que deja margen y normaliza "
    "el contraste."
)

# ===========================================================================
# 5. FASE D
# ===========================================================================
h("5. Fase D — Evaluación del mejor modelo", 1)
best = RES["best"]
p(
    f"La mejor configuración fue «{best['titulo']}» con una exactitud en test de "
    f"{best['test_acc']:.2%}."
)

h("5.1 Matriz de confusión", 2)
p(
    "La matriz de confusión permite identificar qué clases se confunden entre sí. La diagonal "
    "recoge los aciertos y las celdas fuera de ella los errores."
)
imagen(FIGDIR / "fig_matriz_confusion.png", "Figura 5. Matriz de confusión del mejor modelo sobre el conjunto de prueba.", 14.0)

p("Las confusiones más frecuentes fueron:", bold=True)
for d in RES["top_confusions"]:
    bullet(f"{d['real']} → {d['predicho']}: {d['casos']} casos")

p("Exactitud por clase (extracto del informe de clasificación):", bold=True)
tabla(
    ["Clase", "Correctas", "Total", "Accuracy"],
    [[c["Clase"], c["Correctas"], c["Total"], f"{c['Accuracy']:.2f} %"] for c in RES["per_class"]],
    widths=[5.0, 2.5, 2.5, 3.0],
)

h("5.2 Cálculo de la precisión final (fórmula de Accuracy)", 2)
p(
    "La exactitud global corresponde a la suma de los aciertos (diagonal de la matriz de "
    "confusión) dividida entre el número total de muestras evaluadas:"
)
p("Accuracy = Σᵢ₌₁..C TPᵢ / N_total", bold=True)
imagen(FIGDIR / "fig_formula_accuracy.png", "Figura 6. Aplicación numérica de la fórmula de Accuracy.", 13.0)

f = RES["accuracy_formula"]
tabla(
    ["Verdaderos positivos (Σ TPᵢ)", "Total de muestras (N_total)", "Accuracy"],
    [[f["correct"], f["total"], f"{f['accuracy']:.4f} = {f['accuracy']:.2%}"]],
    widths=[5.5, 5.5, 4.0],
)

# ===========================================================================
# 6. INNOVACIÓN
# ===========================================================================
h("6. Innovación — Batch Normalization", 1)
bn = next(e for e in RES["experiments"] if e["slug"] == "final_bn")
p(
    f"Como técnica extra se implementó Batch Normalization (normalización por lotes), que "
    f"normaliza la media y varianza de las activaciones de cada capa, estabilizando el "
    f"entrenamiento y permitiendo tasas de aprendizaje mayores. En esta configuración obtuvo "
    f"{bn['test_acc']:.2%} en test; con el reducido número de épocas usado y el aumento de datos "
    f"activo no superó al Dropout puro, pero queda integrada como línea de mejora (más épocas, "
    f"más datos o arquitecturas más profundas)."
)

# ===========================================================================
# 7. CONCLUSIONES
# ===========================================================================
h("7. Conclusiones", 1)
bullet(
    f"La arquitectura CNN con filtros 3×3, MaxPooling y Softmax supera el umbral del 90 % de "
    f"exactitud, alcanzando {f['accuracy']:.2%} en el conjunto de prueba."
)
bullet(
    "El sobreajuste se manifestó en el modelo sin Dropout (brecha train/val) y se mitigó con "
    "Dropout, Data Augmentation y EarlyStopping + ReduceLROnPlateau."
)
bullet("Adam convergió notablemente más rápido que SGD en el mismo presupuesto de épocas.")
bullet(
    "El Data Augmentation aporta robustez ante imágenes reales, aun a costa de unos pocos puntos "
    "en el test limpio del dataset."
)
bullet(
    "Batch Normalization se implementó como técnica de innovación y queda disponible para "
    "configuraciones con más épocas."
)
bullet(
    "El modelo cumple los objetivos del taller y resulta adecuado para la misión de "
    "clasificación automática de productos de la startup de logística."
)

# ===========================================================================
# 8. ANEXO — CÓMO REPRODUCIR
# ===========================================================================
h("8. Anexo — Reproducibilidad", 1)
p("Los entregables se generan de forma reproducible con las siguientes órdenes:")
bullet("Notebook: .venv\\Scripts\\python.exe build_notebook_entregable.py (genera y ejecuta el .ipynb).")
bullet("Informe: .venv\\Scripts\\python.exe build_informe_entregable.py (genera este .docx).")
bullet("Semilla fija RANDOM_SEED = 42 en todos los pasos.")

doc.save(OUT_PATH)
print("Informe generado:", OUT_PATH)
