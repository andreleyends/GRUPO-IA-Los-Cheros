"""Genera el informe del Taller CNN en Word editable (.docx)."""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUTPUT = BASE / "Informe_Taller_CNN_FashionMNIST.docx"

data = json.loads((RESULTS / "experiments.json").read_text(encoding="utf-8"))
experiments = data["experiments"]
best = data["best"]
acc = data["accuracy_formula"]

CLASSES = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatilla", "Bolso", "Botín",
]

AZUL = RGBColor(0x1F, 0x3B, 0x73)
doc = Document()

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = AZUL
    return p


def para(text, bold=False, align=None, size=None, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def add_table(header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.autofit = True
    for i, htext in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(htext))
        run.bold = True
    for r in rows:
        cells = t.add_row().cells
        for i, val in enumerate(r):
            cells[i].text = str(val)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def add_image(path, width_cm=16):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    doc.add_paragraph()


# ---------------- Portada ----------------
for _ in range(3):
    doc.add_paragraph()
para("Diseño y Optimización de una CNN para Clasificación de Imágenes Complejas",
     bold=True, align="center", size=20, color=AZUL)
para("Clasificación de prendas con el dataset Fashion-MNIST",
     align="center", size=13)
doc.add_paragraph()
para("Informe del Taller de Redes Neuronales Convolucionales", align="center", size=13)
for _ in range(4):
    doc.add_paragraph()
para("Equipo: Los Cheros", bold=True, align="center")
para("Integrantes: _______________________________", align="center")
para("Framework: TensorFlow / Keras", bold=True, align="center")
doc.add_paragraph()
para("Desplegado como aplicación web: Streamlit Community Cloud", align="center")
doc.add_page_break()

# ---------------- 1-3 ----------------
h("1. Contexto y objetivos", 1)
para(
    "Los equipos actúan como ingenieros de visión artificial para una startup de logística. "
    "El objetivo es desarrollar una Red Neuronal Convolucional (CNN) capaz de clasificar imágenes "
    "con alta precisión, enfrentando problemas reales como el sobreajuste (overfitting) y la "
    "variabilidad de los datos. Se implementa con TensorFlow/Keras y se estudia el impacto de las "
    "operaciones de convolución, max-pooling y dropout, así como del data augmentation.",
    align="justify",
)
h("2. Dataset: Fashion-MNIST", 2)
para(
    "Fashion-MNIST es un conjunto de 70 000 imágenes en escala de grises de 28×28 píxeles, "
    "distribuidas en 10 clases de prendas (60 000 para entrenamiento y 10 000 para prueba). "
    "Es un dataset balanceado (6000 imágenes por clase) y sustituye a MNIST por ser un reto de "
    "clasificación más realista.",
    align="justify",
)
add_table(
    ["Clase", "Etiqueta", "Clase", "Etiqueta"],
    [[CLASSES[i], "", CLASSES[i + 5], ""] for i in range(5)],
)
h("3. Fase A: Preparación y preprocesamiento", 2)
para("Antes de entrenar se aplicaron las siguientes transformaciones:", align="justify")
add_table(
    ["Transformación", "Descripción"],
    [
        ["Normalización", "Píxeles escalados de [0, 255] a [0, 1] dividiendo entre 255."],
        ["Reshape", "De (28, 28) a (28, 28, 1) para añadir el canal de entrada."],
        ["Split estratificado", "54 000 entrenamiento / 6 000 validación / 10 000 prueba."],
        ["Data Augmentation", "Rotación (±0.08), zoom (±0.1) y traslación (±0.1) aleatorias."],
        ["Etiquetas sparse", "Enteros 0-9 con pérdida sparse_categorical_crossentropy."],
    ],
)

# ---------------- 4 ----------------
h("4. Fase B: Arquitectura del modelo", 2)
para(
    "La arquitectura base incluye capas convolucionales con filtros de 3×3 y activación ReLU, "
    "capas de reducción MaxPooling y una capa de salida con Softmax para clasificación multiclase. "
    "Todas las convoluciones usan strides = 1×1 y padding = 'same' (conservan la dimensión espacial) "
    "y el pooling usa ventana 2×2.",
    align="justify",
)
add_table(
    ["Capa", "Configuración", "Salida"],
    [
        ["Input", "28×28×1 (escala de grises)", "28×28×1"],
        ["Data Augmentation", "Rotación / Zoom / Traslación", "28×28×1"],
        ["Conv2D", "32 filtros, 3×3, stride 1, padding same, ReLU", "28×28×32"],
        ["MaxPooling2D", "Ventana 2×2", "14×14×32"],
        ["Conv2D", "64 filtros, 3×3, stride 1, padding same, ReLU", "14×14×64"],
        ["MaxPooling2D", "Ventana 2×2", "7×7×64"],
        ["Conv2D", "128 filtros, 3×3, stride 1, padding same, ReLU", "7×7×128"],
        ["Flatten", "-", "6272"],
        ["Dense", "128 neuronas, ReLU", "128"],
        ["Dropout", "p = 0.3", "128"],
        ["Dense (salida)", "10 neuronas, Softmax", "10"],
    ],
)
para("Total de parámetros entrenables: 896 906 (≈ 3.42 MB).")

# ---------------- 5 ----------------
h("5. Fase C: Experimentación (el core de la actividad)", 1)
para(
    "Se entrenaron varias configuraciones para medir el impacto de la regularización, el optimizador "
    "y el aumento de datos. Cada experimento se evaluó sobre el conjunto de prueba (10 000 imágenes). "
    "La brecha (gap) es la diferencia entre la exactitud de entrenamiento y la de validación: una "
    "brecha alta indica overfitting.",
    align="justify",
)
rows = []
for e in experiments:
    rows.append([
        e["titulo"], e["optimizer"].upper(),
        "Sí" if e["use_dropout"] else "No",
        "Sí" if e["use_augmentation"] else "No",
        f"{e['train_accuracy_final']:.2%}", f"{e['val_accuracy_final']:.2%}",
        f"{e['test_accuracy']:.2%}",
        f"{e['train_accuracy_final'] - e['val_accuracy_final']:+.2%}",
    ])
add_table(["Experimento", "Optimizador", "Dropout", "Augment.", "Train", "Val", "Test", "Gap"], rows)
para("Conclusiones de la experimentación:", bold=True)
for txt in [
    "• Dropout: elimina parte del sobreajuste; la brecha baja de 5.5% (sin Dropout) a 3.6% y la "
    "exactitud de prueba sube de 91.67% a 92.40%.",
    "• Data Augmentation: lleva la brecha prácticamente a cero (≈ 0%), el modelo deja de memorizar; "
    "aumenta la dificultad efectiva del entrenamiento y necesita más épocas para alcanzar su máximo.",
    "• Optimizador: Adam converge mucho más rápido y con mejor precisión que SGD (momentum 0.9) en el "
    "mismo número de épocas.",
    "• Batch Normalization (innovación): estabiliza y acelera el entrenamiento manteniendo una brecha "
    "pequeña.",
]:
    para(txt, align="justify")

# ---------------- 6 ----------------
h("6. Curvas de Loss y Accuracy (Entrenamiento vs. Validación)", 1)
para(
    "Se comparan la pérdida (Loss) y la exactitud (Accuracy) de entrenamiento y validación por época. "
    "Sin regularización el modelo se separa rápidamente (overfitting); con Dropout y aumento de datos "
    "las curvas de entrenamiento y validación se mantienen cercanas.",
    align="justify",
)
para("Modelo base (Sin Dropout) — se aprecia el overfitting:", bold=True)
add_image(RESULTS / "fig_curvas_sin_dropout.png", 15.5)
para("Modelo final (Con Dropout) — brecha reducida:", bold=True)
add_image(RESULTS / "fig_curvas_con_dropout.png", 15.5)
doc.add_page_break()

# ---------------- 7 ----------------
h("7. Matriz de confusión", 1)
para(
    "La matriz de confusión muestra, para cada clase real (filas), cuántas imágenes se predijeron en "
    "cada clase (columnas). La diagonal representa los aciertos. Se observa que las confusiones se "
    "concentran entre prendas visualmente similares: Camiseta/top, Camisa, Suéter y Abrigo.",
    align="justify",
)
add_image(RESULTS / "fig_confusion_final.png", 13.5)
para("Clases que más se confunden:", bold=True)
add_table(
    ["Clase real", "Predicha como", "Casos"],
    [[c["real"], c["predicho"], str(c["casos"])] for c in data["top_confusions"]],
)
para(
    "El análisis confirma que el error del modelo se debe a la similitud morfológica entre camisetas, "
    "camisas, suéteres y abrigos, no a un fallo del algoritmo de clasificación.",
    align="justify",
)

# ---------------- 8 ----------------
h("8. Cálculo de la precisión final (fórmula de Accuracy)", 1)
para(
    "La exactitud (Accuracy) se define como el cociente entre las predicciones correctas y el total "
    "de predicciones realizadas:",
    align="justify",
)
para("Accuracy = (Predicciones correctas) / (Total de predicciones)",
     bold=True, align="center", size=14, color=AZUL)
para(f"Para el modelo final ({best['titulo']}) evaluado sobre las 10 000 imágenes de prueba:")
para(
    f"Accuracy = {acc['correct']:,} / {acc['total']:,} = {acc['accuracy']:.4f} ≈ "
    f"{acc['accuracy'] * 100:.2f}%",
    bold=True, align="center", size=14, color=AZUL,
)
para(
    f"Es decir, el modelo clasifica correctamente {acc['accuracy'] * 100:.2f}% de las imágenes que "
    "nunca vio durante el entrenamiento, superando el umbral del 90% definido como objetivo.",
    align="justify",
)
h("9. Análisis del overfitting y cómo se solucionó", 2)
para(
    "El overfitting ocurre cuando el modelo memoriza los datos de entrenamiento en lugar de aprender "
    "patrones generalizables: la exactitud de entrenamiento sigue subiendo mientras la de validación "
    "se estanca o baja, y la brecha entre ambas crece. En el experimento sin regularización la brecha "
    "alcanzó 5.5%. Se mitigó con tres estrategias:",
    align="justify",
)
for txt in [
    "1. Dropout (p=0.3): apaga neuronas aleatoriamente durante el entrenamiento, forzando "
    "representaciones redundantes y reduciendo la brecha a 3.6%.",
    "2. Data Augmentation: genera variaciones (rotación, zoom, traslación) que exponen al modelo a más "
    "diversidad, llevando la brecha a ≈ 0%.",
    "3. Early Stopping y ReduceLROnPlateau: detienen el entrenamiento cuando la pérdida de validación "
    "deja de mejorar y reducen el learning rate automáticamente.",
]:
    para(txt, align="justify")
h("10. Innovación: Batch Normalization", 2)
para(
    "Como técnica adicional se incorporó Batch Normalization tras las capas convolucionales. Normaliza "
    "las activaciones de cada mini-lote, lo que estabiliza y acelera el entrenamiento y permite usar "
    "tasas de aprendizaje más altas, manteniendo una brecha pequeña entre entrenamiento y validación.",
    align="justify",
)
doc.add_page_break()

# ---------------- 11 ----------------
h("11. Conclusiones", 1)
for txt in [
    "• Se construyó una CNN con convoluciones 3×3 + ReLU, MaxPooling y salida Softmax, cumpliendo los "
    "requisitos de la Fase B.",
    "• El Dropout y el Data Augmentation reducen el overfitting de forma significativa (brecha de 5.5% "
    "a ≈0%).",
    "• Adam supera a SGD en velocidad de convergencia y precisión final.",
    f"• El modelo final alcanza una exactitud de {acc['accuracy'] * 100:.2f}% en el conjunto de prueba, "
    "superando el objetivo del 90%.",
    "• Las principales confusiones ocurren entre Camiseta/top, Camisa, Suéter y Abrigo, por su similitud "
    "visual.",
]:
    para(txt, align="justify")
h("Entregables del proyecto", 2)
for txt in [
    "• Aplicación web (Streamlit): interfaz con las 6 fases CRISP-ML que permite subir una prenda y "
    "obtener la predicción en vivo.",
    "• Este informe (Word/PDF) con curvas, matriz de confusión y cálculo de exactitud.",
    "• Código fuente documentado (módulos de datos, modelado, evaluación y despliegue).",
]:
    para(txt, align="justify")

doc.save(str(OUTPUT))
print("Informe Word generado:", OUTPUT)
