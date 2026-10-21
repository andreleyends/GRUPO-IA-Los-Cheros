"""Genera el informe PDF del Taller CNN a partir de results/experiments.json.

Salida: Informe_Taller_CNN_FashionMNIST.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUTPUT = BASE / "Informe_Taller_CNN_FashionMNIST.pdf"

data = json.loads((RESULTS / "experiments.json").read_text(encoding="utf-8"))
experiments = data["experiments"]
best = data["best"]
acc = data["accuracy_formula"]
hparams = data["hparams"]

CLASSES = [
    "Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatilla", "Bolso", "Botín",
]

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17, textColor=colors.HexColor("#1F3B73"), spaceAfter=10)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#2E5AAC"), spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=10, alignment=TA_JUSTIFY, leading=14)
TITLE = ParagraphStyle("TITLE", parent=styles["Title"], fontSize=24, textColor=colors.HexColor("#1F3B73"))
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
CENTER = ParagraphStyle("CENTER", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER)


def P(text):
    return Paragraph(text, BODY)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2 * cm, 1.2 * cm, "Taller CNN - Fashion-MNIST")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.restoreState()


def table(data, col_widths=None, font=9, header_bg="#1F3B73"):
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#AAAAAA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


def image(path, width_cm=16):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = width_cm * cm * ratio
    return img


story = []

# ---------------- Portada ----------------
story += [
    Spacer(1, 3 * cm),
    Paragraph("Diseño y Optimización de una CNN para Clasificación de Imágenes Complejas", TITLE),
    Spacer(1, 0.5 * cm),
    Paragraph("Clasificación de prendas con el dataset <b>Fashion-MNIST</b>", SUB),
    Spacer(1, 1.5 * cm),
    Paragraph("Informe del Taller de Redes Neuronales Convolucionales", SUB),
    Spacer(1, 2 * cm),
    Paragraph("<b>Equipo:</b> Los Cheros", CENTER),
    Paragraph("<b>Integrantes:</b> _______________________________", CENTER),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Framework:</b> TensorFlow / Keras", CENTER),
    Spacer(1, 3 * cm),
    Paragraph("Desplegado como aplicación web: Streamlit Community Cloud", CENTER),
    PageBreak(),
]

# ---------------- 1. Contexto ----------------
story += [
    Paragraph("1. Contexto y objetivos", H1),
    P(
        "Los equipos actúan como ingenieros de visión artificial para una startup de logística. "
        "El objetivo es desarrollar una <b>Red Neuronal Convolucional (CNN)</b> capaz de clasificar "
        "imágenes con alta precisión, enfrentando problemas reales como el <b>sobreajuste (overfitting)</b> "
        "y la variabilidad de los datos. Se implementa con TensorFlow/Keras y se estudia el impacto de "
        "las operaciones de <b>convolución</b>, <b>max-pooling</b> y <b>dropout</b>, así como del "
        "<b>data augmentation</b>."
    ),
    Paragraph("2. Dataset: Fashion-MNIST", H2),
    P(
        "Fashion-MNIST es un conjunto de 70 000 imágenes en escala de grises de 28×28 píxeles, "
        "distribuidas en <b>10 clases</b> de prendas (60 000 para entrenamiento y 10 000 para prueba). "
        "Es un dataset balanceado (6000 imágenes por clase) y sustituye a MNIST por ser un reto de "
        "clasificación más realista."
    ),
    table(
        [["Clase", "Etiqueta", "Clase", "Etiqueta"]] +
        [[CLASSES[i], "", CLASSES[i + 5], ""] for i in range(5)],
        col_widths=[2.2 * cm, 4.3 * cm, 2.2 * cm, 4.3 * cm],
        font=8.5,
    ),
    Spacer(1, 0.3 * cm),
    Paragraph("3. Fase A: Preparación y preprocesamiento", H2),
    P("Antes de entrenar se aplicaron las siguientes transformaciones:"),
    table(
        [
            ["Transformación", "Descripción"],
            ["Normalización", "Píxeles escalados de [0, 255] a [0, 1] dividiendo entre 255."],
            ["Reshape", "De (28, 28) a (28, 28, 1) para añadir el canal de entrada."],
            ["Split estratificado", "54 000 entrenamiento / 6 000 validación / 10 000 prueba."],
            ["Data Augmentation", "Rotación (±0.08), zoom (±0.1) y traslación (±0.1) aleatorias."],
            ["Etiquetas sparse", "Enteros 0-9 con pérdida sparse_categorical_crossentropy."],
        ],
        col_widths=[4 * cm, 12 * cm],
        font=9,
    ),
]

# ---------------- 4. Arquitectura ----------------
story += [
    Paragraph("4. Fase B: Arquitectura del modelo", H2),
    P(
        "La arquitectura base incluye capas convolucionales con filtros de <b>3×3</b> y activación "
        "<b>ReLU</b>, capas de reducción <b>MaxPooling</b> y una capa de salida con <b>Softmax</b> "
        "para clasificación multiclase. Todas las convoluciones usan <b>strides = 1×1</b> y "
        "<b>padding = 'same'</b> (conservan la dimensión espacial) y el pooling usa ventana 2×2."
    ),
    table(
        [
            ["Capa", "Configuración", "Salida"],
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
        col_widths=[3.2 * cm, 9.5 * cm, 3.3 * cm],
        font=8.5,
    ),
    Spacer(1, 0.2 * cm),
    Paragraph("Total de parámetros entrenables: <b>896 906</b> (≈ 3.42 MB).", BODY),
]

# ---------------- 5. Fase C ----------------
story += [
    Paragraph("5. Fase C: Experimentación (el core de la actividad)", H1),
    P(
        "Se entrenaron varias configuraciones para medir el impacto de la regularización, el optimizador "
        "y el aumento de datos. Cada experimento se evaluó sobre el conjunto de prueba (10 000 imágenes). "
        "La brecha (gap) es la diferencia entre la exactitud de entrenamiento y la de validación: una "
        "brecha alta indica overfitting."
    ),
]
exp_rows = [["Experimento", "Optimizador", "Dropout", "Augment.", "Train", "Val", "Test", "Gap"]]
for e in experiments:
    exp_rows.append([
        e["titulo"], e["optimizer"].upper(),
        "Sí" if e["use_dropout"] else "No",
        "Sí" if e["use_augmentation"] else "No",
        f"{e['train_accuracy_final']:.2%}", f"{e['val_accuracy_final']:.2%}",
        f"{e['test_accuracy']:.2%}", f"{e['train_accuracy_final'] - e['val_accuracy_final']:+.2%}",
    ])
story += [table(exp_rows, font=8), Spacer(1, 0.3 * cm)]
story += [
    P(
        "<b>Conclusiones de la experimentación:</b>"
    ),
    P(
        "• <b>Dropout:</b> elimina parte del sobreajuste: la brecha baja de 5.5% (sin Dropout) a 3.6% "
        "y la exactitud de prueba sube de 91.67% a 92.40%.<br/>"
        "• <b>Data Augmentation:</b> lleva la brecha prácticamente a cero (≈ 0%), es decir, el modelo "
        "deja de memorizar; aumenta la dificultad efectiva del entrenamiento, por lo que necesita más "
        "épocas para alcanzar su máximo.<br/>"
        "• <b>Optimizador:</b> <b>Adam</b> converge mucho más rápido y con mejor precisión que <b>SGD</b> "
        "(con momentum 0.9) en el mismo número de épocas.<br/>"
        "• <b>Batch Normalization</b> (innovación) estabiliza y acelera el entrenamiento, manteniendo una "
        "brecha pequeña."
    ),
]

# ---------------- 6. Curvas ----------------
story += [
    Paragraph("6. Curvas de Loss y Accuracy (Entrenamiento vs. Validación)", H1),
    P(
        "Se comparan la pérdida (Loss) y la exactitud (Accuracy) de entrenamiento y validación por época. "
        "Sin regularización el modelo se separa rápidamente (overfitting); con Dropout y aumento de datos "
        "las curvas de entrenamiento y validación se mantienen cercanas."
    ),
    Paragraph("<b>Modelo base (Sin Dropout) — se aprecia el overfitting:</b>", BODY),
    image(RESULTS / "fig_curvas_sin_dropout.png", 15.5),
    Spacer(1, 0.4 * cm),
    Paragraph("<b>Modelo final (Con Dropout) — brecha reducida:</b>", BODY),
    image(RESULTS / "fig_curvas_con_dropout.png", 15.5),
    PageBreak(),
]

# ---------------- 7. Matriz de confusión ----------------
story += [
    Paragraph("7. Matriz de confusión", H1),
    P(
        "La matriz de confusión muestra, para cada clase real (filas), cuántas imágenes se predijeron "
        "en cada clase (columnas). La diagonal representa los aciertos. Se observa que las confusiones "
        "se concentran entre prendas visualmente similares: <b>Camiseta/top, Camisa, Suéter y Abrigo</b>."
    ),
    image(RESULTS / "fig_confusion_final.png", 13.5),
    Spacer(1, 0.3 * cm),
    Paragraph("<b>Clases que más se confunden:</b>", BODY),
]
conf_rows = [["Clase real", "Predicha como", "Casos"]]
for c in data["top_confusions"]:
    conf_rows.append([c["real"], c["predicho"], str(c["casos"])])
story += [table(conf_rows, col_widths=[5 * cm, 5 * cm, 3 * cm], font=9), Spacer(1, 0.4 * cm)]
story += [
    P(
        "El análisis confirma que el error del modelo se debe a la similitud morfológica entre camisetas, "
        "camisas, suéteres y abrigos, no a un fallo del algoritmo de clasificación."
    ),
    PageBreak(),
]

# ---------------- 8. Accuracy ----------------
story += [
    Paragraph("8. Cálculo de la precisión final (fórmula de Accuracy)", H1),
    P(
        "La exactitud (Accuracy) se define como el cociente entre las predicciones correctas y el total "
        "de predicciones realizadas:"
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        "<b>Accuracy = (Predicciones correctas) / (Total de predicciones)</b>",
        ParagraphStyle("F", parent=CENTER, fontSize=13, textColor=colors.HexColor("#1F3B73")),
    ),
    Spacer(1, 0.4 * cm),
    P(
        f"Para el modelo final (<b>{best['titulo']}</b>) evaluado sobre las 10 000 imágenes de prueba:"
    ),
    Spacer(1, 0.2 * cm),
    Paragraph(
        f"<b>Accuracy = {acc['correct']:,} / {acc['total']:,} = {acc['accuracy']:.4f} ≈ "
        f"{acc['accuracy'] * 100:.2f}%</b>",
        ParagraphStyle("F2", parent=CENTER, fontSize=13, textColor=colors.HexColor("#1F3B73")),
    ),
    Spacer(1, 0.5 * cm),
    P(
        "Es decir, el modelo clasifica correctamente <b>"
        f"{acc['accuracy'] * 100:.2f}%</b> de las imágenes que nunca vio durante el entrenamiento, "
        "superando el umbral del 90% definido como objetivo."
    ),
    Paragraph("9. Análisis del overfitting y cómo se solucionó", H2),
    P(
        "El <b>overfitting</b> ocurre cuando el modelo memoriza los datos de entrenamiento en lugar de "
        "aprender patrones generalizables: la exactitud de entrenamiento sigue subiendo mientras la de "
        "validación se estanca o baja, y la brecha entre ambas crece. En el experimento sin regularización "
        "la brecha alcanzó 5.5%. Se mitigó con tres estrategias:"
    ),
    P(
        "1. <b>Dropout (p=0.3):</b> apaga neuronas aleatoriamente durante el entrenamiento, forzando "
        "representaciones redundantes y reduciendo la brecha a 3.6%.<br/>"
        "2. <b>Data Augmentation:</b> genera variaciones (rotación, zoom, traslación) que exponen al "
        "modelo a más diversidad, llevando la brecha a ≈ 0%.<br/>"
        "3. <b>Early Stopping y ReduceLROnPlateau:</b> detienen el entrenamiento cuando la pérdida de "
        "validación deja de mejorar y reducen el learning rate automáticamente."
    ),
    Paragraph("10. Innovación: Batch Normalization", H2),
    P(
        "Como técnica adicional se incorporó <b>Batch Normalization</b> tras las capas convolucionales. "
        "Normaliza las activaciones de cada mini-lote, lo que estabiliza y acelera el entrenamiento y "
        "permite usar tasas de aprendizaje más altas, manteniendo una brecha pequeña entre entrenamiento "
        "y validación."
    ),
    PageBreak(),
]

# ---------------- 11. Conclusiones ----------------
story += [
    Paragraph("11. Conclusiones", H1),
    P(
        "• Se construyó una CNN con convoluciones 3×3 + ReLU, MaxPooling y salida Softmax, cumpliendo los "
        "requisitos de la Fase B.<br/>"
        "• El <b>Dropout</b> y el <b>Data Augmentation</b> reducen el overfitting de forma significativa "
        "(brecha de 5.5% a ≈0%).<br/>"
        "• <b>Adam</b> supera a <b>SGD</b> en velocidad de convergencia y precisión final.<br/>"
        f"• El modelo final alcanza una exactitud de <b>{acc['accuracy'] * 100:.2f}%</b> en el conjunto "
        "de prueba, superando el objetivo del 90%.<br/>"
        "• Las principales confusiones ocurren entre Camiseta/top, Camisa, Suéter y Abrigo, por su "
        "similitud visual."
    ),
    Spacer(1, 0.5 * cm),
    Paragraph("<b>Entregables del proyecto</b>", H2),
    P(
        "• <b>Aplicación web (Streamlit):</b> interfaz con las 6 fases CRISP-ML que permite subir una "
        "prenda y obtener la predicción en vivo.<br/>"
        "• <b>Este informe PDF</b> con curvas, matriz de confusión y cálculo de exactitud.<br/>"
        "• <b>Código fuente</b> documentado (módulos de datos, modelado, evaluación y despliegue)."
    ),
]

doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    topMargin=2 * cm,
    bottomMargin=2 * cm,
    leftMargin=2 * cm,
    rightMargin=2 * cm,
    title="Informe Taller CNN - Fashion-MNIST",
    author="Equipo Los Cheros",
)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print("Informe generado:", OUTPUT)
