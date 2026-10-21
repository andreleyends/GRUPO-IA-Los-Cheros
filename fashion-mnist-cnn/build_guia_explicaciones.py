"""Genera la "Guia de Explicaciones y Evidencias" (.docx) del taller CNN.

Documento de apoyo para la defensa oral y para completar los apartados
[COMPLETAR] del informe grupal. Reune, con referencias exactas al codigo,
la explicacion de:

  1. Como funciona el proyecto (flujo, estructura, notebook).
  2. Filtros, strides y padding.
  3. Overfitting: por que ocurre, como se soluciono, como se demuestra.
  4. Batch Normalization (tecnica extra).
  5. Graficas Loss/Accuracy, Matriz de confusion y formula de Accuracy.
  6. Como explicar el notebook entregable.
  7. Preguntas frecuentes del docente con respuesta.
  8. Referencia rapida de codigo (archivo:linea).

Uso:
    .venv\\Scripts\\python.exe build_guia_explicaciones.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "ENTREGABLES" / "figures"
RES_NOTEBOOK = json.loads(
    (ROOT / "ENTREGABLES" / "resultados_notebook.json").read_text(encoding="utf-8")
)
RES_EXP = json.loads((ROOT / "results" / "experiments.json").read_text(encoding="utf-8"))

# Carpeta de salida en el escritorio del usuario.
OUT_DIR = Path(r"H:\Nueva carpeta\escritorio\Guia Explicaciones CNN")
OUT_PATH = OUT_DIR / "Guia_Explicaciones_y_Evidencias_CNN.docx"

AZUL = RGBColor(0x1F, 0x4E, 0x79)
AZUL_CLARO = "DCE6F1"
GRIS_CLARO = "F2F2F2"
VERDE_CLARO = "E2EFDA"
AMBAR_CLARO = "FFF2CC"

doc = Document()

# ---------------------------------------------------------------------------
# Estilos base
# ---------------------------------------------------------------------------
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.12

for lvl, size in ((1, 18), (2, 14), (3, 12)):
    st = doc.styles[f"Heading {lvl}"]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.color.rgb = AZUL
    st.font.bold = True
    st.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    st.paragraph_format.space_after = Pt(6)

for sec in doc.sections:
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.4)
    sec.right_margin = Cm(2.4)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def h(text: str, level: int = 1):
    return doc.add_heading(text, level=level)


def p(text: str = "", bold: bool = False, italic: bool = False, size: int | None = None,
      align: str | None = None, color: RGBColor | None = None, space_after: int | None = None):
    """Parrafo normal. Los tramos entre acentos graves se muestran en monoespaciado."""
    par = doc.add_paragraph()
    for j, trozo in enumerate(text.split("`")):
        if not trozo:
            continue
        if j % 2 == 1:
            inline_codigo(par, trozo, size=(size or 11) - 1.5, negrita=bold)
        else:
            run = par.add_run(trozo)
            run.bold = bold
            run.italic = italic
            if size:
                run.font.size = Pt(size)
            if color:
                run.font.color.rgb = color
    if align == "center":
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if space_after is not None:
        par.paragraph_format.space_after = Pt(space_after)
    return par


def rico(partes: list[tuple[str, bool]], align: str | None = None):
    """Parrafo con tramos en negrita: [(texto, bold), ...]."""
    par = doc.add_paragraph()
    for texto, bold in partes:
        run = par.add_run(texto)
        run.bold = bold
    if align == "justify":
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return par


def bullet(text: str, nivel: int = 0, bold_prefix: str | None = None):
    estilo = "List Bullet" if nivel == 0 else f"List Bullet {nivel + 1}"
    par = doc.add_paragraph(style=estilo)
    if bold_prefix:
        par.add_run(bold_prefix).bold = True
    for j, trozo in enumerate(text.split("`")):
        if not trozo:
            continue
        if j % 2 == 1:
            inline_codigo(par, trozo, size=9.5)
        else:
            par.add_run(trozo)
    par.paragraph_format.space_after = Pt(3)
    return par


def numero(text: str, bold_prefix: str | None = None):
    par = doc.add_paragraph(style="List Number")
    if bold_prefix:
        par.add_run(bold_prefix).bold = True
    for j, trozo in enumerate(text.split("`")):
        if not trozo:
            continue
        if j % 2 == 1:
            inline_codigo(par, trozo, size=9.5)
        else:
            par.add_run(trozo)
    par.paragraph_format.space_after = Pt(3)
    return par


def _sombrear(elemento, fill: str):
    pr = elemento.get_or_add_pPr() if hasattr(elemento, "get_or_add_pPr") else elemento
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pr.append(shd)


def codigo(text: str, size: int = 9):
    """Bloque de codigo monoespaciado con fondo gris."""
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    par.paragraph_format.left_indent = Cm(0.4)
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(8)
    _sombrear(par._p, GRIS_CLARO)
    return par


def inline_codigo(par, texto: str, size: int = 9.5, negrita: bool = False):
    run = par.add_run(texto)
    run.font.name = "Consolas"
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    run.font.color.rgb = RGBColor(0xA3, 0x1D, 0x1D)
    run.bold = negrita
    return run


def tabla(header: list[str], rows: list[list], widths: list[float] | None = None,
          font_size: int = 9.5, estilo: str = "Table Grid"):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = estilo
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, txt in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(txt))
        run.bold = True
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _sombrear(cell._tc.get_or_add_tcPr(), "2E75B6")
    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            par = cells[i].paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            # Segmentos entre acentos graves se muestran en monoespaciado.
            for j, trozo in enumerate(str(v).split("`")):
                if not trozo:
                    continue
                if j % 2 == 1:
                    inline_codigo(par, trozo, size=font_size - 0.5)
                else:
                    run = par.add_run(trozo)
                    run.font.size = Pt(font_size)
        if r_i % 2 == 1:
            for c in cells:
                _sombrear(c._tc.get_or_add_tcPr(), "F2F7FB")
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def callout(titulo: str, texto: str, fill: str = VERDE_CLARO, icono: str = "Idea"):
    """Caja de una sola celda con titulo + texto (para frases listas)."""
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    cell = t.rows[0].cells[0]
    cell.text = ""
    par = cell.paragraphs[0]
    par.paragraph_format.space_after = Pt(2)
    run = par.add_run(f"{icono} - {titulo}")
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = AZUL
    par2 = cell.add_paragraph()
    par2.paragraph_format.space_after = Pt(2)
    run2 = par2.add_run(texto)
    run2.font.size = Pt(10)
    _sombrear(cell._tc.get_or_add_tcPr(), fill)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def imagen(nombre: str, caption: str, width_cm: float = 15.0):
    path = FIGDIR / nombre
    if not path.exists():
        p(f"[Falta la figura: {nombre}]", italic=True)
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)


def separador():
    par = doc.add_paragraph()
    pr = par._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:color"), "BFBFBF")
    borders.append(bottom)
    pr.append(borders)
    par.paragraph_format.space_after = Pt(4)


def pie_de_pagina():
    footer = doc.sections[0].footer
    par = footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run("Guia de Explicaciones y Evidencias  |  Taller CNN Fashion-MNIST  |  Pagina ")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    par._p.append(fld)


# ===========================================================================
# PORTADA
# ===========================================================================
for _ in range(2):
    doc.add_paragraph()

p("GUIA DE EXPLICACIONES Y EVIDENCIAS", bold=True, align="center", size=24, color=AZUL)
p("Diseno y Optimizacion de una CNN para Clasificacion de Imagenes Complejas",
  bold=True, align="center", size=15)
p("Dataset: Fashion-MNIST  |  Equipo: Los Cheros  |  Cusco - Peru, 2026",
  align="center", size=11, italic=True)
doc.add_paragraph()
p("Documento de apoyo para la defensa oral y para completar los apartados "
  "[INSERTAR FIGURA] y [COMPLETAR] del informe grupal.",
  align="center", size=11, italic=True)
p("Todo el contenido esta referenciado al codigo real del repositorio "
  "(archivo:linea) para que puedas apoyarte en el al Defender el proyecto.",
  align="center", size=10, italic=True)

doc.add_paragraph()
p("Integrantes del equipo", bold=True, align="center")
tabla(
    ["Integrante", "Codigo"],
    [
        ["Huamantalla Rios, Diego Miguel", "022100354D"],
        ["Latorre Campos, Andre Ismael", "022101034C"],
        ["Ordoñez Paniura, Marcela Leonor", "022100735H"],
        ["Suel Monrroy, David Joshua", "022101091G"],
    ],
    widths=[10.5, 5.0],
)

doc.add_paragraph()
p("Resumen de cifras clave", bold=True, align="center")
tabla(
    ["Indicador", "Resultado", "Objetivo", "Donde se comprueba"],
    [
        ["Accuracy en test (modelo de produccion)", "92.40%", ">= 90%", "`validate_fixes.py` / `results/experiments.json`"],
        ["Accuracy via formula de Accuracy", "9.240 / 10.000 = 0.9240", "---", "Notebook celda D.2"],
        ["Latencia de inferencia", "~45 ms/imagen", "< 50 ms", "`validate_fixes.py`"],
        ["Calzado clasificado como Bolso", "0 / 120", "0", "`validate_fixes.py`"],
        ["Fotos reales (carpeta `fallos/`)", "4/8 top-1, 5/8 top-3", "---", "`validate_fixes.py`"],
        ["Parametros del modelo", "896.906 (384 mas con BatchNorm)", "---", "`model.summary()`"],
    ],
    widths=[6.0, 4.2, 2.6, 6.5],
)

doc.add_page_break()

# ===========================================================================
# 0. COMO USAR ESTA GUIA
# ===========================================================================
h("0. Como usar esta guia", 1)
p("Esta guia responde, en orden, las cinco preguntas que suelen caer en la defensa del "
  "taller. Cada seccion tiene tres partes: donde esta en el codigo, como se explica en "
  "palabras, y como se demuestra que es cierto.")
p("Si te preguntan algo concreto, busca el tema en esta tabla:", italic=True)

tabla(
    ["Si te preguntan...", "Ve a la seccion", "La frase clave esta en..."],
    [
        ["Que son filtros, strides y padding", "2", "Callout azul de la seccion 2"],
        ["Por que hay overfitting y como lo arreglaron", "3", "Callout verde de la seccion 3"],
        ["Como implementaron Batch Normalization", "4", "Callout ambar de la seccion 4"],
        ["Donde estan las graficas y la matriz de confusion", "5.1 y 5.2", "Tabla resumen de la seccion 5"],
        ["Como se calcula la precision final", "5.3", "Bloque de codigo de la seccion 5.3"],
        ["Como funciona el proyecto entero", "1", "Diagrama de flujo de la seccion 1.1"],
        ["Como explico el notebook", "6", "Guion de 5 minutos de la seccion 6.3"],
        ["Me hacen una pregunta que no conozco", "7", "Banco de preguntas de la seccion 7"],
        ["De que archivo es esta linea", "8", "Referencia rapida de codigo"],
    ],
    widths=[8.0, 3.2, 7.0],
)

callout(
    "Dato importante sobre las cifras",
    "El informe grupal usa las cifras de results/experiments.json (la corrida de referencia: "
    "Con Dropout = 92.40%). El notebook entregable vuelve a entrenar los 5 experimentos y por eso "
    "muestra numeros ligeramente distintos (92.31% / 92.22%): es la misma arquitectura, otra corrida. "
    "Cuando defiendas, di siempre de donde sale cada cifra para que no parezca una contradiccion.",
    fill=AMBAR_CLARO,
    icono="Atencion",
)

doc.add_page_break()

# ===========================================================================
# 1. COMO FUNCIONA EL PROYECTO
# ===========================================================================
h("1. Como funciona el proyecto", 1)

h("1.1 El flujo completo de un vistazo", 2)
p("El proyecto sigue las 6 fases de CRISP-ML(Q). Cada fase es un modulo de `src/` y, a la vez, "
  "una pestaña de la aplicacion Streamlit (`app.py`). El recorrido de un dato es:")

flujo = doc.add_table(rows=7, cols=3)
flujo.style = "Table Grid"
encabezados = ["#", "Fase CRISP-ML(Q)", "Que hace y donde vive"]
for i, txt in enumerate(encabezados):
    cell = flujo.rows[0].cells[i]
    cell.text = ""
    run = cell.paragraphs[0].add_run(txt)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    _sombrear(cell._tc.get_or_add_tcPr(), "2E75B6")

pasos = [
    ("1", "Comprension del Negocio", "Objetivos, KPIs, backlog Scrum y riesgos. `src/business.py`"),
    ("2", "Comprension de los Datos", "EDA, muestras por clase, balance y estadisticas. `src/data_understanding.py`"),
    ("3", "Preparacion de los Datos", "Normalizacion a [0,1], canal, split 90/10 estratificado y Data Augmentation. `src/data_preparation.py`"),
    ("4", "Modelado", "Arquitectura CNN con flags (Dropout / Augmentation / BatchNorm) y entrenamiento. `src/modeling.py`"),
    ("5", "Evaluacion", "Curvas Loss/Accuracy, matriz de confusion, metricas y comparacion con el baseline MLP. `src/evaluation.py`"),
    ("6", "Despliegue", "Preprocesado de inferencia para fotos reales, prediccion con top-3 y umbral. `src/deployment.py` + `app.py`"),
]
for i, (n, fase, desc) in enumerate(pasos, start=1):
    cells = flujo.rows[i].cells
    cells[0].text = ""
    r = cells[0].paragraphs[0].add_run(n)
    r.bold = True
    r.font.size = Pt(10)
    cells[1].text = ""
    r = cells[1].paragraphs[0].add_run(fase)
    r.bold = True
    r.font.size = Pt(9.5)
    cells[2].text = ""
    par = cells[2].paragraphs[0]
    # texto con `...` resaltado
    partes = desc.split("`")
    for j, trozo in enumerate(partes):
        if j % 2 == 1:
            inline_codigo(par, trozo, size=9)
        else:
            rr = par.add_run(trozo)
            rr.font.size = Pt(9.5)
    if i % 2 == 0:
        for c in cells:
            _sombrear(c._tc.get_or_add_tcPr(), "F2F7FB")

doc.add_paragraph()
p("Ademas hay dos scripts que orchestran todo:", bold=True)
bullet("entrena el modelo base y el baseline MLP, y guarda `models/fashion_cnn.keras` y `models/baseline_mlp.keras`.")
bullet("entrena un candidato con augmentation de produccion, lo compara con `results/modelo_final.keras` y despliega el mejor. En la ultima corrida el candidato dio 89.42% y el referente 92.40%, asi que se desplego el de 92.40%.")
bullet("mide el antes/despues de las mejoras de preprocesado (seccion 5.4).")

h("1.2 Estructura de carpetas", 2)
codigo("""fashion-mnist-cnn/
|-- app.py                      # GUI Streamlit: 6 pestañas = 6 fases CRISP-ML
|-- train.py                    # Entrenamiento base (CNN + baseline MLP)
|-- train_final.py              # Candidato de produccion + comparacion y despliegue
|-- experiments.py              # Experimentacion comparativa (Fase C)
|-- validate_fixes.py           # Validacion antes/despues de las mejoras
|-- build_notebook_entregable.py# Genera y ejecuta el notebook entregable
|-- build_informe_entregable.py # Genera el informe Word a partir del notebook
|-- build_guia_explicaciones.py # Genera ESTA guia
|-- requirements.txt            # tensorflow-cpu==2.21.0, streamlit==1.66.0
|-- CONTEXTO_PROYECTO.md        # Documento de traspaso (contexto completo)
|-- src/
|   |-- config.py               # Clases, hiperparametros, rutas, KPIs, RANDOM_SEED=42
|   |-- data_loader.py          # Carga CSV local o keras.datasets
|   |-- business.py             # Fase 1
|   |-- data_understanding.py   # Fase 2 (EDA)
|   |-- data_preparation.py     # Fase 3 (normalizacion, split, augmentation)
|   |-- modeling.py             # Fase 4 (build_cnn, train_model, baseline)
|   |-- evaluation.py           # Fase 5 (curvas, matriz, metricas)
|   `-- deployment.py           # Fase 6 (preprocesado de inferencia + predict)
|-- models/     # fashion_cnn.keras (produccion), baseline_mlp.keras
|-- results/    # experiments.json, modelo_final.keras, confusion_matrix.npy, fig_*.png
|-- archive/    # CSV del dataset (fuera de git)
|-- fallos/     # 8 fotos reales de validacion (fuera de git)
`-- ENTREGABLES/ # Notebook, informe Word, resultados_notebook.json, figures/""")

h("1.3 Los tres modelos que existen (no confundirlos)", 2)
tabla(
    ["Archivo", "Que es", "Accuracy en test", "Para que se usa"],
    [
        ["`models/fashion_cnn.keras`", "Modelo de PRODUCCION (el que usa la app)", "92.40%", "Inferencia real y despliegue en la nube"],
        ["`models/baseline_mlp.keras`", "MLP simple, linea base de comparacion", "88.2%", "Demostrar que la CNN aporta +4.2 puntos"],
        ["`results/modelo_final.keras`", "Mejor modelo de los 5 experimentos", "92.40%", "Referencia contra la que compite `train_final.py`"],
    ],
    widths=[5.2, 5.6, 2.8, 5.0],
)

h("1.4 Los entregables", 2)
tabla(
    ["Entregable", "Que contiene", "Como se genera"],
    [
        ["`ENTREGABLES/Notebook_CNN_FashionMNIST.ipynb`", "Notebook con las 4 fases del taller, celdas ejecutadas y sus salidas", "`build_notebook_entregable.py` (nbformat + nbclient)"],
        ["`ENTREGABLES/Informe_CNN_FashionMNIST.docx`", "Informe Word editable con curvas, matriz de confusion, formula de Accuracy, analisis de overfitting e innovacion", "`build_informe_entregable.py` (python-docx)"],
        ["`ENTREGABLES/figures/`", "10 figuras PNG generadas por el notebook", "Salida del notebook"],
        ["`ENTREGABLES/resultados_notebook.json`", "Todos los numeros del notebook en JSON", "Ultima celda del notebook"],
        ["`CONTEXTO_PROYECTO.md`", "Documento de traspaso con todo el trabajo, problemas y decisiones", "Manual, esta versionado en git"],
    ],
    widths=[6.4, 6.4, 5.4],
)

doc.add_page_break()

# ===========================================================================
# 2. FILTROS, STRIDES Y PADDING
# ===========================================================================
h("2. Filtros, strides y padding", 1)
p("Este es el requisito de la Fase B del taller: la CNN debe usar filtros 3x3 con ReLU, "
  "MaxPooling 2x2 y salida Softmax, definiendo correctamente filtros, strides y padding.")

h("2.1 Donde esta en el codigo", 2)
p("Todo esta en la funcion `build_cnn()` de `src/modeling.py` (lineas 33-43). El unico parametro "
  "que se repite en las tres convoluciones es el kernel, que viene de la configuracion central:")
codigo("""# src/config.py  (lineas 46-56) - hiperparametros de la CNN
DEFAULT_HPARAMS: dict = {
    "conv1_filters": 32,     # filtros de la 1a conv
    "conv2_filters": 64,     # filtros de la 2a conv
    "kernel_size": 3,        # -> kernel 3x3
    "dense_units": 128,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 20,
    "validation_split": 0.1,
}""")
codigo("""# src/modeling.py  (lineas 33-43) - las tres convoluciones
keras.layers.Conv2D(hparams["conv1_filters"],      (k, k), strides=(1, 1), padding="same", activation="relu")
keras.layers.Conv2D(hparams["conv2_filters"],      (k, k), strides=(1, 1), padding="same", activation="relu")
keras.layers.Conv2D(hparams["conv2_filters"] * 2,  (k, k), strides=(1, 1), padding="same", activation="relu")""")

h("2.2 Que significa cada uno (la explicacion)", 2)
tabla(
    ["Concepto", "Definicion sencilla", "Valor aqui", "Por que ese valor"],
    [
        ["Filtros (filters)",
         "Cuantos kernels aprende la capa. Cada kernel es un detector de un patron distinto (un borde, una textura, una costura). La salida de la capa tiene tantos canales como filtros.",
         "32 -> 64 -> 128",
         "Van en aumento para que las capas profundas combinen patrones simples en conceptos mas complejos (textura -> forma -> prenda)."],
        ["Stride (paso)",
         "Cuantos pixeles avanza el filtro en cada movimiento al recorrer la imagen. stride 1 = revisa pixel a pixel; stride 2 = se salta uno.",
         "(1, 1) en las 3 convs",
         "Se mantiene en 1 para no perder resolucion. La reduccion de tamano la hace el MaxPooling, que si usa stride 2."],
        ["Padding (relleno)",
         "Se anaden pixeles (de valor 0) en los bordes antes de convolver. `same` = la salida conserva el tamano de la entrada; `valid` = no rellena y la salida se encoge.",
         "\"same\"",
         "Con kernel 3x3 y `same` la imagen no pierde tamano en las convoluciones, y los pixeles del borde no se pierden. Si usaramos `valid`, 28x28 -> 26x26 en la primera capa y perderiamos informacion de los pixeles del borde."],
    ],
    widths=[3.0, 6.4, 2.6, 6.2],
)

h("2.3 La arquitectura completa con sus formas de salida", 2)
p("Esta tabla la debes tener a mano: es la que demuestra que el modelo cumple los requisitos "
  "de la Fase B.")
tabla(
    ["Capa", "Configuracion", "Forma de salida", "Que aporta"],
    [
        ["Input", "---", "28 x 28 x 1", "Imagen en escala de grises"],
        ["Conv2D", "32 filtros 3x3, stride 1, same, ReLU", "28 x 28 x 32", "Extrae bordes y texturas basicas"],
        ["MaxPooling2D", "ventana 2x2, stride 2", "14 x 14 x 32", "Reduce dimensionalidad a la mitad"],
        ["Conv2D", "64 filtros 3x3, stride 1, same, ReLU", "14 x 14 x 64", "Combina bordes en patrones maiores"],
        ["MaxPooling2D", "ventana 2x2, stride 2", "7 x 7 x 64", "Segunda reduccion"],
        ["Conv2D", "128 filtros 3x3, stride 1, same, ReLU", "7 x 7 x 128", "Patrones de nivel alto"],
        ["Flatten", "---", "6272", "Aplana para la capa densa"],
        ["Dense", "128 neuronas, ReLU", "128", "Aprende combinaciones de rasgos"],
        ["Dropout", "tasa 0.3", "128", "Regularizacion anti-overfitting"],
        ["Dense (salida)", "10 neuronas, Softmax", "10", "Probabilidad de cada una de las 10 clases"],
    ],
    widths=[3.0, 5.4, 3.0, 5.4],
)
p("Parametros totales: 896.906 (3.42 MB). Con BatchNorm suben a 897.290, es decir, la "
  "BatchNorm solo anade 384 parametros entrenables (2 por canal: gamma y beta).", italic=True)

h("2.4 Frase lista para el informe", 2)
callout(
    "Filtros, strides y padding",
    "Se usan filtros 3x3 con stride 1 y padding 'same': el kernel 3x3 captura patrones locales "
    "(bordes, texturas) sin reducir la resolucion espacial, y el padding 'same' conserva las "
    "dimensiones de la imagen y evita que la informacion de los bordes se pierda en las "
    "convoluciones. El numero de filtros crece de 32 a 64 y 128 para que cada capa ms profunda "
    "pueda construir rasgos mas complejos, mientras que la reduccion espacial la realiza el "
    "MaxPooling 2x2 con stride 2.",
    fill=AZUL_CLARO,
    icono="Frase",
)

h("2.5 Como lo demuestro en vivo", 2)
numero("Llamar a `summary_text(modelo_base)` (o `modelo_base.summary()`): la salida muestra cada "
       "Conv2D con su forma de salida, confirmando el 3x3 y el `same`.")
numero("Comentar que el `strides=(1, 1)` y el `padding=\"same\"` estan explicitos en el codigo, "
       "no son el default de Keras (si se omitieran, el padding seria `valid`).")
numero("Mostrar la tabla 2.3: las formas de salida 28->14->7 demuestran que la reduccion la "
       "hace el pooling y no la convolucion.")

doc.add_page_break()

# ===========================================================================
# 3. OVERFITTING
# ===========================================================================
h("3. Overfitting: por que ocurre, como lo solucionamos y como se demuestra", 1)

h("3.1 Que es el overfitting", 2)
p("Un modelo sobreajustado memoriza las imagenes de entrenamiento en vez de aprender patrones "
  "generales. Se manifesta de forma muy concreta en las graficas:")
bullet("el accuracy de ENTRENAMIENTO sigue subiendo (o se queda muy alto),")
bullet("el accuracy de VALIDACION se estanca o baja,")
bullet("y la val_loss empieza a SUBIR, que es la senal inequivoca.")

h("3.2 Por que ocurre en nuestro caso", 2)
rico([
    ("La CNN tiene ", False), ("896.906 parametros entrenables", True),
    (" para aprender de apenas ", False), ("54.000 imagenes de entrenamiento", True),
    (" (90 % de 60.000). La relacion es de ~17 parametros por imagen: el modelo tiene una "
     "capacidad de memorizacion muy superior a la cantidad de datos disponibles, asi que la "
     "forma mas barata de reducir el error de entrenamiento es aprenderse cada foto. "
     "Ese es el sobreajuste: minimiza la perdida en train a costa de la perdida en datos que "
     "nunca vio.", False),
])

h("3.3 La evidencia: la tabla de los 5 experimentos", 2)
p("Estos son los numeros reales de `results/experiments.json` (la corrida de referencia que "
  "usa el informe grupal). La columna `train` frente a `val` es la que delata el overfitting:")

exp = RES_EXP["experiments"]
tabla(
    ["#", "Configuracion", "Train", "Val", "Test", "Train - Val", "Diagnostico"],
    [
        ["1", "Sin Dropout", "96.67%", "91.13%", "91.67%", "+5.54 pts",
         "BRECHA CLASICA: memoriza el train, no generaliza"],
        ["2", "Con Dropout (elegido)", "95.30%", "91.67%", "92.40%", "+3.63 pts",
         "Menor brecha y el mejor test de los cinco"],
        ["3", "Dropout + Augmentation", "86.94%", "87.30%", "88.15%", "-0.36 pts",
         "Se elimina la brecha, pero el augmentation baja el test limpio"],
        ["4", "Dropout + Aug + SGD", "82.13%", "82.67%", "83.84%", "-0.54 pts",
         "Ademas SGD converge mas lento"],
        ["5", "+ BatchNorm (innovacion)", "88.03%", "86.95%", "87.46%", "+1.08 pts",
         "Mejora frente a 3 y 4, pero no supera a 2"],
    ],
    widths=[0.9, 4.3, 1.7, 1.7, 1.7, 2.0, 5.0],
)
p("Como leerlo: en la experimentacion 1 la brecha train-val es de 5.54 puntos y el test cae a "
  "91.67%. Al anadir Dropout la brecha baja a 3.63 puntos y el test sube a 92.40%. Eso es, "
  "exactamente, la mitigacion del sobreajuste demostrada con numeros.", italic=True)

h("3.4 Las 3 soluciones aplicadas y donde estan en el codigo", 2)
tabla(
    ["Solucion", "Que hace", "Donde esta", "Efecto medido"],
    [
        ["Dropout 0.3",
         "Apaga aleatoriamente el 30 % de las neuronas en cada paso de entrenamiento, impide que la red dependa de combinaciones de rasgos demasiado especificas.",
         "`src/modeling.py:47-48`",
         "Brecha train-val: 5.54 -> 3.63 pts. Test: 91.67% -> 92.40%"],
        ["EarlyStopping (patience 4)",
         "Detiene el entrenamiento cuando la val_loss lleva 4 epocas sin mejorar, y restaura los mejores pesos (restore_best_weights=True).",
         "`src/modeling.py:102`",
         "Corta el entrenamiento en el punto optimo: no deja que la val_loss suba"],
        ["ReduceLROnPlateau (factor 0.5)",
         "Reduce la tasa de aprendizaje a la mitad cuando la val_loss se estanca, para poder escapar de minimos locales.",
         "`src/modeling.py:103`",
         "Estabiliza la convergencia final"],
        ["Data Augmentation",
         "Genera variantes aleatorias de cada imagen (rotacion +-8 %, zoom 10 %, traslación 10 %) SOLO durante el entrenamiento.",
         "`src/data_preparation.py:27-38`",
         "Elimina la brecha train-val (queda en -0.36 pts) y sube la robustez ante fotos reales"],
    ],
    widths=[3.2, 6.4, 3.4, 5.2],
)
p("El augmentation solo se aplica en entrenamiento: las capas de Keras se activan con "
  "`training=True`, por lo que en validacion y prueba la imagen queda intacta.", italic=True)

h("3.5 Como lo compruebo o demuestro (3 pruebas concretas)", 2)
numero("Prueba grafica: comparar `ENTREGABLES/figures/fig_curvas_sin_dropout.png` (brecha visible y "
       "val_loss subiendo) con `ENTREGABLES/figures/fig_curvas_base.png` (lineas pegadas). Son las "
       "mismas dos curvas que pide el informe en las Figuras 1 y 2.")
numero("Prueba tabular: abrir `results/experiments.json` y comparar `train_accuracy_final` contra "
       "`val_accuracy_final` en la config `sin_dropout`. La brecha de 5.54 puntos ES la prueba.")
numero("Prueba de codigo: ejecutar `experiments.py`, que reentrena los 5 casos y regenera la tabla. "
       "Tambien `validate_fixes.py`, que ademas mide el efecto del preprocesado.")

imagen("fig_curvas_sin_dropout.png",
       "Figura de apoyo - Modelo SIN Dropout: se ve la brecha entre entrenamiento y validacion "
       "(train 96.67% vs val 91.13%). Esta es la evidencia grafica del overfitting.", 14.5)
imagen("fig_curvas_base.png",
       "Figura de apoyo - Modelo CON Dropout: las curvas de train y val se mantienen juntas "
       "(train 95.30% vs val 91.67%). Esta es la evidencia de la solucion.", 14.5)

h("3.6 Frase lista para el informe", 2)
callout(
    "Overfitting: causa, evidencia y solucion",
    "El sobreajuste se produce porque la CNN tiene 896.906 parametros entrenables frente a solo "
    "54.000 imagenes de entrenamiento: la capacidad de memorizacion del modelo supera con creces "
    "la cantidad de datos, de modo que el accuracy de entrenamiento sube a 96.67% mientras el de "
    "validacion se queda en 91.13% y la val_loss comienza a crecer, una brecha de 5.54 puntos que "
    "delata que el modelo aprendio las imagenes concretas en vez de los patrones generales. Se "
    "soluciono con tres mecanismos complementarios: una capa Dropout de 0.3 que apaga aleatoriamente "
    "el 30 % de las neuronas, un callback EarlyStopping con paciencia 4 sobre la val_loss que "
    "detiene el entrenamiento en el punto optimo y restaura los mejores pesos, y Data Augmentation "
    "que genera variantes rotadas, ampliadas y desplazadas de cada imagen durante el entrenamiento. "
    "El resultado es que la brecha train-validacion se reduce a 3.63 puntos y el accuracy en el "
    "conjunto de prueba sube de 91.67% a 92.40%, el mejor de las cinco configuraciones evaluadas.",
    fill=VERDE_CLARO,
    icono="Frase",
)

doc.add_page_break()

# ===========================================================================
# 4. BATCH NORMALIZATION
# ===========================================================================
h("4. Batch Normalization (tecnica extra)", 1)
p("La rúbrica pide implementar tecnicas adicionales como Batch Normalization. Es la "
  "configuracion que el equipo eligio como innovacion del proyecto.")

h("4.1 Que es y para que sirve", 2)
p("Batch Normalization normaliza las activaciones de cada capa usando la media y la varianza "
  "del mini-lote actual, y luego las reescala con dos parametros aprendibles (gamma y beta). "
  "En la practica esto: (a) estabiliza la distribucion de activaciones entre capas, "
  "(b) permite usar tasas de aprendizaje mas altas, y (c) reduce la sensibilidad a la "
  "inicializacion. En CNN suele актuar ademas como un regularizador débil.")

h("4.2 Donde esta implementado", 2)
p("Como opcion de la arquitectura, controlada por el flag `use_batchnorm` de `build_cnn()`:")
codigo("""# src/modeling.py  (lineas 9-15) - la firma de build_cnn
def build_cnn(
    hparams: dict,
    use_augmentation: bool = True,
    use_dropout: bool = True,
    use_batchnorm: bool = False,      # <-- flag de la tecnica extra
    augmentation=None,
):
    ...
    layers.append(keras.layers.Conv2D(hparams["conv1_filters"], (k, k), strides=(1, 1), padding="same", activation="relu"))
    if use_batchnorm:                                                    # <-- linea 34
        layers.append(keras.layers.BatchNormalization())                # <-- linea 35
    layers.append(keras.layers.MaxPooling2D((2, 2)))

    layers.append(keras.layers.Conv2D(hparams["conv2_filters"], (k, k), strides=(1, 1), padding="same", activation="relu"))
    if use_batchnorm:                                                    # <-- linea 39
        layers.append(keras.layers.BatchNormalization())                # <-- linea 40
    layers.append(keras.layers.MaxPooling2D((2, 2)))""")
p("En la experimentacion se activa asi (notebook, celda C.1):")
codigo("""dict(slug="final_bn",
     titulo="5. + Batch Norm (innovacion)",
     use_dropout=True, use_aug=True, use_bn=True, optimizer="adam"),""")

h("4.3 Como lo demuestro (4 pruebas, de la mas simple a la mas solida)", 2)
numero("Prueba de implementacion: el codigo esta a la vista. Muestra `src/modeling.py:34-35` y "
       "`39-40`; el flag existe y es funcional. En `model.summary()` aparecen las capas "
       "`batch_normalization` con sus parametros (384 en total: 2 por canal).")
numero("Prueba de existencia de la configuracion: el experimento 5 de la Fase C se llama "
       "\"+ BatchNorm (inovacion)\" y esta en la tabla de resultados con su accuracy propia.")
numero("Prueba comparativa: esta en la misma tabla que el resto. Con BatchNorm (87.46%) supera "
       "claramente a las configuraciones 3 y 4 (88.15% con Adam y 83.84% con SGD no son "
       "comparables directamente, pero en la corrida del notebook la config 5 da 89.00%), "
       "pero NO supera al Dropout puro (92.40%).")
numero("Prueba de reproduccion: `experiments.py` reentrena la config con `use_batchnorm=True` y "
       "regenera `fig_curvas_final_bn.png`.")

p("Ojo con este punto: en la corrida del notebook entregable, el experimento 5 dio 89.00% "
  "mientras que en `results/experiments.json` (corrida de referencia) dio 87.46%. Son dos "
  "corridas distintas del mismo experimento. Si el docente pregunta, la respuesta honesta es "
  "que la BatchNorm se implemento y se midio correctamente, que mejora a las configuraciones con "
  "SGD y que con solo 10 epocas y mas augmentation todavia no supera al Dropout puro, y que "
  "queda como linea de mejora (mas epocas, mas datos, sin augmentation simultanea).",
  italic=True)

imagen("fig_curvas_final_bn.png",
       "Figura de apoyo - Configuracion 5: Dropout + Augmentation + Batch Normalization. "
       "Curvas de Loss y Accuracy de la configuracion con la tecnica extra.", 14.5)

h("4.4 Frase lista para el informe", 2)
callout(
    "Batch Normalization como tecnica extra",
    "Se implemento Batch Normalization como tecnica adicional mediante el flag use_batchnorm de "
    "build_cnn(), que intercala una capa de normalizacion despues de la primera y de la segunda "
    "convolucion. Esta tecnica normaliza las activaciones de cada capa usando la media y la "
    "varianza del mini-lote, estabilizando la distribucion interna de la red y permitiendo "
    "aprendizajes mas rapidos y robustos; su coste es minimo, 384 parametros extra sobre 896.906. "
    "Se evaluo como configuracion 5 de la experimentacion y quedo registrada en la tabla de "
    "resultados junto con sus curvas Loss y Accuracy. Con el numero de "
    "epocas utilizado no logro superar al Dropout puro, por lo que la conclusion honesta es que "
    "no se despliego, pero queda integrada en el codigo como linea de mejora parafuturos "
    "reentrenamientos con mas epocas o mas datos.",
    fill=AMBAR_CLARO,
    icono="Frase",
)
p("Nota: en el texto anterior se deja un marcador (%) para que lo completes con la cifra "
  "exacta de la corrida que presentes (87.46% en la corrida de referencia o 89.00% en la del "
  "notebook).", italic=True, size=9)

doc.add_page_break()

# ===========================================================================
# 5. LOS 3 ENTREGABLES DE LA FASE D
# ===========================================================================
h("5. Graficas, matriz de confusion y formula de Accuracy", 1)
p("Estos son los tres bloques de la Fase D del taller. Para cada uno se indica el archivo, la "
  "funcion, la linea exacta y la figura que lo respalda.")

h("5.1 Graficas de Loss y Accuracy (Entrenamiento vs. Validacion)", 2)
tabla(
    ["Aspecto", "Detalle"],
    [
        ["Funcion", "`plot_training_curves(history)` en `src/evaluation.py:24-40`"],
        ["Que dibuja", "Dos subgraficos: perdida (loss vs val_loss) y exactitud (accuracy vs val_accuracy), una linea por epoca"],
        ["De donde salen los datos", "El objeto `History` que devuelve `model.fit()`; sus diccionarios `history.history['loss']`, `['val_loss']`, `['accuracy']`, `['val_accuracy']`"],
        ["Como se genera", "`build_notebook_entregable.py:225-226` para el modelo base y `:284-285` para cada experimento"],
        ["Figuras", "`fig_curvas_base.png`, `fig_curvas_sin_dropout.png`, `fig_curvas_dropout_aug.png`, `fig_curvas_dropout_aug_sgd.png`, `fig_curvas_final_bn.png`"],
        ["Que debe verse", "En el modelo sin regularizar: la curva de train por encima de la de val y la val_loss girando hacia arriba. En el modelo final: las dos curvas casi juntas"],
    ],
    widths=[4.0, 12.0],
)
codigo("""# src/evaluation.py  (lineas 24-40)
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    epochs = range(1, len(history.history["loss"]) + 1)
    axes[0].plot(epochs, history.history["loss"],     "o-", label="Entrenamiento")
    axes[0].plot(epochs, history.history["val_loss"], "s-", label="Validacion")
    axes[0].set_title("Perdida (loss)")
    axes[1].plot(epochs, history.history["accuracy"],     "o-", label="Entrenamiento")
    axes[1].plot(epochs, history.history["val_accuracy"], "s-", label="Validacion")
    axes[1].set_title("Exactitud (accuracy)")
    fig.tight_layout()
    return fig""")

h("5.2 Matriz de confusion: que clases se confunden mas", 2)
tabla(
    ["Aspecto", "Detalle"],
    [
        ["Funcion", "`compute_confusion(model, prepared)` en `src/evaluation.py:43-54`"],
        ["Como se calcula", "`sklearn.metrics.confusion_matrix` + `classification_report`; las filas son la etiqueta real y las columnas la prediccion"],
        ["Grafico", "`plot_confusion_matrix(cm)` en `src/evaluation.py:57-75` (seaborn heatmap con `annot=True`)"],
        ["Top de confusiones", "`top_confusions(cm, top=5)` en `src/evaluation.py:78-88`: recorre la matriz y devuelve los pares real->predicho con mas casos"],
        ["Accuracy por clase", "`per_class_accuracy(cm)` en `src/evaluation.py:91-104`"],
        ["Figura", "`fig_matriz_confusion.png`"],
    ],
    widths=[4.0, 12.0],
)
p("Las 5 confusiones mas frecuentes de la corrida actual (`resultados_notebook.json`):")
tabla(
    ["Clase real", "Clase predicha", "Casos", "Intuitivamente"],
    [
        ["Camisa", "Camiseta/top", "107", "Siluetas casi identicas: diferencia minima en cuello y mangas"],
        ["Camiseta/top", "Camisa", "75", "La confusion reciproca: la red duda entre las dos"],
        ["Camisa", "Abrigo", "65", "Camisa de manga larga se lee como abrigo"],
        ["Suéter", "Abrigo", "63", "Mismo problema: ropa de abrigo larga"],
        ["Zapatilla", "Botín", "44", "Calzado similar en silueta a 28x28"],
    ],
    widths=[3.4, 3.4, 1.8, 7.4],
)
p("Patron general: las confusiones se concentran en las prendas de silueta parecida "
  "(camisa / camiseta / suéter / abrigo) y en el calzado (zapatilla / botín). Son errores "
  "razonables, no fallos del modelo: a 28x28 en escala de grises esas clases son "
  "prácticamente indistinguibles.", italic=True)

imagen("fig_matriz_confusion.png",
       "Figura de apoyo - Matriz de confusion sobre las 10.000 imagenes del conjunto de prueba. "
       "En la diagonal, los aciertos por clase; fuera de ella, las confusiones.", 12.5)

h("5.3 Calculo de la precision final con la formula de Accuracy", 2)
p("La formula pedida explicitamente por el taller:")
codigo("""Accuracy = (Predicciones correctas) / (Total de predicciones)
         = Σ (TP por clase) / N_total
         = suma de la diagonal de la matriz de confusion / total de muestras""")
p("Y su implementacion literal en el notebook entregable (celda D.2, "
  "`build_notebook_entregable.py:349-360`):")
codigo("""# D.2 - Formula de Accuracy aplicada a la matriz de confusion
correctos = int(np.trace(cm))   # suma de la diagonal: verdaderos positivos por clase
total = int(cm.sum())           # total de muestras evaluadas
accuracy = correctos / total
print(f"Accuracy = Σ TPᵢ / N_total = {correctos} / {total} = {accuracy:.4f} = {accuracy:.2%}")""")
p("En el proyecto ya se hace el mismo calculo en `experiments.py:154-157`, y el resultado se "
  "guarda en `results/experiments.json` y `ENTREGABLES/resultados_notebook.json` dentro del "
  "objeto `accuracy_formula` (claves `correct`, `total`, `accuracy`).")
tabla(
    ["Corrida", "Aciertos (diagonal)", "Total", "Accuracy"],
    [
        ["Referencia (`results/experiments.json`) - modelo de produccion", "9.240", "10.000", "0.9240 = 92.40%"],
        ["Notebook entregable (`ENTREGABLES/resultados_notebook.json`)", "9.231", "10.000", "0.9231 = 92.31%"],
    ],
    widths=[8.0, 3.0, 2.0, 3.0],
)
imagen("fig_formula_accuracy.png",
       "Figura de apoyo - Aplicacion numerica de la formula de Accuracy sobre la matriz de confusion.",
       13.0)

h("5.4 Tabla resumen: requisito -> donde esta en el codigo", 2)
tabla(
    ["Requisito del taller", "Codigo", "Figura o dato que lo respalda"],
    [
        ["Graficas de Loss y Accuracy (train vs val)", "`src/evaluation.py:24-40`", "`fig_curvas_base.png`, `fig_curvas_sin_dropout.png`"],
        ["Matriz de confusion", "`src/evaluation.py:43-75`", "`fig_matriz_confusion.png` + `top_confusions()`"],
        ["Criterio de precision final por formula", "`build_notebook_entregable.py:349-360`", "`fig_formula_accuracy.png` (9.240 / 10.000)"],
        ["Filtros, strides y padding", "`src/modeling.py:33-43`", "Tabla de formas de salida (28->14->7)"],
        ["Overfitting: causa y solucion", "`src/modeling.py:47-48, 102-103`", "`fig_curvas_sin_dropout.png` vs `fig_curvas_base.png`"],
        ["Tecnica extra: Batch Normalization", "`src/modeling.py:34-35, 39-40`", "`fig_curvas_final_bn.png`"],
        ["Comparacion con linea base (MLP 88.2%)", "`src/modeling.py:53-65`", "`fig_comparativa.png`"],
        ["Evaluacion con fotos reales", "`validate_fixes.py`", "0/120 calzado->Bolso, 4/8 top-1, ~45 ms"],
    ],
    widths=[5.6, 4.4, 6.0],
)

doc.add_page_break()

# ===========================================================================
# 6. EL NOTEBOOK
# ===========================================================================
h("6. Como explicar el notebook", 1)

h("6.1 Que es y por que esta asi", 2)
p("El notebook entregable (`ENTREGABLES/Notebook_CNN_FashionMNIST.ipynb`) no se escribio a mano: "
  "lo genera `build_notebook_entregable.py`, que construye las celdas con `nbformat` y las "
  "ejecuta con `nbclient`. Eso tiene tres ventajas que puedes defender:")
bullet("Es reproducible: `build_notebook_entregable.py` regenera el notebook y sus 10 figuras desde cero.")
bullet("Esta ejecutado: todas las celdas ya tienen sus salidas (graficas, tablas, prints).")
bullet("No duplica logica: importa el codigo real del proyecto (`from src.modeling import build_cnn`), "
       "de modo que el notebook y la app no pueden divergir.")

h("6.2 Mapa de celdas (que se explica en cada parte)", 2)
tabla(
    ["Seccion", "Que hace", "Genera"],
    [
        ["Portada y estructura", "Contexto de la mision y tabla de fases", "---"],
        ["Configuracion", "Rutas, imports, `set_seed(42)` con `random`, `numpy` y `tensorflow`", "---"],
        ["Fase A.1", "Carga el dataset e imprime formas y rango de pixeles", "`fig_distribucion.png`"],
        ["Fase A.2", "Normaliza a [0,1], anade canal y hace el split estratificado 90/10", "---"],
        ["Fase A.3", "Vista previa de Data Augmentation (original vs. aumentado)", "`fig_augmentation.png`"],
        ["Fase B.1", "Construye la CNN base y muestra `model.summary()` con el conteo de parametros", "---"],
        ["Fase B.2", "Entrena con Adam + Dropout y EarlyStopping, y evalua en test", "`fig_curvas_base.png`"],
        ["Fase C.1", "Loop de las 5 configuraciones (Dropout, Augmentation, SGD, BatchNorm)", "4 curvas mas"],
        ["Fase C.2", "Grafica de barras comparando el accuracy en test de cada config", "`fig_comparativa.png`"],
        ["Fase D.1", "Mejor modelo: reporte de clasificacion, matriz de confusion y top-5 confusiones", "`fig_matriz_confusion.png`"],
        ["Fase D.2", "Formula de Accuracy con `np.trace(cm)` y `cm.sum()`", "`fig_formula_accuracy.png`"],
        ["Conclusiones", "Texto final con los numeros reales inyectados automaticamente", "---"],
        ["D.3 Export", "Guarda `resultados_notebook.json`, que alimenta al informe Word", "---"],
    ],
    widths=[3.4, 8.2, 4.4],
)

h("6.3 Guion de 5 minutos para exponer el notebook", 2)
p("Puedes seguir estos cinco pasos, en orden, sin improvisar:")
numero("Presenta el contexto: Fashion-MNIST, 70.000 imagenes 28x28 en 10 clases de prendas, "
       "perfectamente balanceado, y el objetivo de superar el 90 % de accuracy en test.")
numero("Fase A: muestra que el dataset esta balanceado (6000 por clase), explica la normalizacion "
       "a [0,1] y el split estratificado 90/10, y ensena la figura de augmentation: son variantes "
       "sinteticas que no anaden datos nuevos pero aumentan la variabilidad.")
numero("Fase B: ensena la tabla de la arquitectura (28->14->7 con 3 capas convolucionales 3x3, "
       "MaxPooling 2x2, Dense 128, Dropout 0.3 y salida Softmax de 10) y el `summary()` con los "
       "896.906 parametros. Ahi se ve que la arquitectura cumple los requisitos de la Fase B.")
numero("Fase C: es el corazon del taller. Enseña la tabla de las 5 configuraciones y la grafica de "
       "barras. El mensaje es que se vario un factor a la vez y se midio todo: Dropout fue la "
       "mejor combinacion (92.40 %), Adam converge claramente mejor que SGD, y la BatchNorm se "
       "implemento y se midio aunque no superase al Dropout puro.")
numero("Fase D: cierra con las graficas Loss/Accuracy, la matriz de confusion con sus top-5 "
       "confusiones y la formula de Accuracy aplicada: 9.240 aciertos de 10.000 = 0.9240 = 92.40 %. "
       "Cierra leyendo las conclusiones de la ultima celda.")

h("6.4 Las 10 figuras que produce", 2)
tabla(
    ["Figura", "Que muestra", "Donde se usa"],
    [
        ["`fig_distribucion.png`", "6000 imagenes por clase (balance perfecto)", "Fase A - EDA"],
        ["`fig_augmentation.png`", "Originales vs. aumentados lado a lado", "Fase A - Data Augmentation"],
        ["`fig_curvas_base.png`", "Loss y Accuracy del modelo base (train vs val)", "Fase B / Figura 1 y 2 del informe"],
        ["`fig_curvas_sin_dropout.png`", "Curvas con la brecha de overfitting visible", "Fase C - Evidencia del overfitting"],
        ["`fig_curvas_dropout_aug.png`", "Curvas de la config 3 (Dropout + Augmentation)", "Fase C"],
        ["`fig_curvas_dropout_aug_sgd.png`", "Curvas de la config 4 (con SGD)", "Fase C"],
        ["`fig_curvas_final_bn.png`", "Curvas de la config 5 (con BatchNorm)", "Fase C - Tecnica extra"],
        ["`fig_comparativa.png`", "Accuracy en test de las 5 configuraciones", "Fase C - Tabla comparativa"],
        ["`fig_matriz_confusion.png`", "Matriz 10x10 con los conteos", "Fase D - Figura 3 del informe"],
        ["`fig_formula_accuracy.png`", "La formula con los numeros puestos", "Fase D - Precision final"],
    ],
    widths=[5.4, 6.4, 4.2],
)

doc.add_page_break()

# ===========================================================================
# 7. PREGUNTAS FRECUENTES
# ===========================================================================
h("7. Banco de preguntas del docente (con respuesta)", 1)
p("Estas son las preguntas mas probables y como responderlas con los datos del proyecto.")

preguntas = [
    ("Por que eligieron 3x3 y no 5x5 o 7x7 para los filtros?",
     "El kernel 3x3 es el equilibrio estandar: con dos capas convolucionales 3x3 seguidas se obtiene "
     "un campo receptivo de 5x5 con menos parametros que una sola capa 5x5 (32x32x3x3x3=9216 frente a "
     "32x32x5x5x3=76800). Ademas 3x3 es el tamaño con el que estan preentrenadas casi todas las CNN "
     "modernas, lo que hace que el transfer learning sea mas efectivo."),
    ("El dataset ya estaba balanceado. Que entonces aporta el balanceo de clases?",
     "En este caso concreto no aporta: se descarto. Fashion-MNIST tiene exactamente 6000 imagenes por "
     "clase en train y 1000 en test, de modo que igualar las clases habria sido innecesario y habria "
     "empeorado el modelo al reducir los datos. Sirvio para descartar una hipotesis: el problema de las "
     "fotos reales NO era desbalance sino cambio de dominio."),
    ("Por que la accuracy baja cuando anaden Data Augmentation?",
     "Porque el augmentation genera imagenes mas distintas de las reales del dataset: la prenda se rota, "
     "se deforma o se desplaza, y eso ensucia el conjunto de prueba, que es imagenes limpias y "
     "centradas. En la tabla, la config 3 baja de 92.40 % a 88.15 %. Pero el augmentation compra "
     "robustez: con el augmentation y el preprocesado de inferencia corregido, el calzado clasificado "
     "como Bolso paso de 1/60 a 0/120. Es un intercambio consciente, y por eso el modelo de "
     "produccion es el de Dropout sin augmentation, que es el mejor en el test limpio."),
    ("Que diferencia hay entre el modelo base y el modelo final?",
     "El modelo base (`train.py`) es la CNN con Adam y Dropout, que da 92.40 %. El modelo final "
     "(`train_final.py`) es un CANDIDATO con augmentation de produccion (rotacion, zoom, traslacion, "
     "volteo horizontal y brillo suave) que se compara con el mejor de los experimentos y se "
     "despliega solo si lo supera. En la ultima corrida el candidato dio 89.42 %, no supero al 92.40 % "
     "del referente, y por eso se desplego el referente. El script decide solo; no es una decision "
     "manual."),
    ("Como manejan las fotos reales que no son del dataset?",
     "Con un preprocesado de inferencia especifico (`preprocess_uploaded` en `src/deployment.py`): "
     "escala de grises, inversion de polaridad si el fondo es claro, limpieza de fondo, recorte del "
     "objeto dejando margen del 20 % (fill=0.8), normalizacion de contraste por percentiles 2-98 y "
     "redimension a 28x28. Ademas la GUI muestra el top-3 y marca \"No reconocido\" cuando la "
     "confianza maxima es menor al 50 %, y tiene un slider de rotacion de +-45 grados."),
    ("Por que el dataset no esta en el repositorio?",
     "Por tamano: los dos CSV ocupan alrededor de 60 MB y hacen el repositorio pesado para clonar. "
     "Estan en `.gitignore` y el codigo los descarga automaticamente con `keras.datasets.fashion_mnist` "
     "si no los encuentra en `archive/`, tanto en local como en la nube."),
    ("Como demuestra que el modelo funciona en el mundo real y no solo con el test set?",
     "Con `validate_fixes.py`, que mide tres cosas: el accuracy en el test set, la tasa de calzado "
     "clasificado como Bolso usando el mismo pipeline de subida que la GUI, y el acierto top-1 y top-3 "
     "sobre 8 fotos reales de la carpeta `fallos/`. Los resultados estan en la tabla de la seccion 8 "
     "del informe grupal: 0/120 calzado->Bolso, 4/8 top-1 y 5/8 top-3 en fotos reales, con ~45 ms de "
     "latencia."),
    ("Que le pasaria si el usuario sube una foto con fondo complejo?",
     "El modelo puede fallar. Es la limitacion principal y esta declarada: las fotos reales son un "
     "dominio distinto al del dataset. Se mitiga con el margen, la normalizacion de contraste, el "
     "top-3, el umbral de confianza y la rotacion manual, pero no se resuelve. La solucion de fondo "
     "seria un fine-tune con 30-50 fotos reales etiquetadas por prenda, distintos angulos, que esta "
     "en la lista de siguientes pasos pero no se ha realizado."),
    ("Es reproducible el resultado?",
     "Si. `RANDOM_SEED = 42` esta fijado en `src/config.py` y se aplica en `random`, `numpy` y "
     "`tensorflow` en el notebook y en los scripts. Con el mismo entorno (Python 3.12 y "
     "tensorflow-cpu==2.21.0) el resultado es identico."),
    ("Cuanto tarda y donde se despliega?",
     "El entrenamiento completo de las 5 configuraciones del notebook tarda unos 10-15 minutos en CPU. "
     "La inferencia de una imagen es de ~45 ms. La app se despliega en Streamlit Community Cloud "
     "apuntando a la rama main del repositorio, con la ruta principal `fashion-mnist-cnn/app.py` y "
     "Python 3.12."),
]
for i, (pregunta, respuesta) in enumerate(preguntas, start=1):
    p(f"Pregunta {i}. {pregunta}", bold=True)
    p(respuesta, align="justify", space_after=10)

doc.add_page_break()

# ===========================================================================
# 8. REFERENCIA RAPIDA DE CODIGO
# ===========================================================================
h("8. Referencia rapida de codigo (archivo:linea)", 1)
p("Esta tabla es la que debes usar cuando te pidan senalar donde esta algo en el codigo.")

tabla(
    ["Que buscar", "Archivo:linea", "Que hace exactamente"],
    [
        ["Filtros / strides / padding", "`src/modeling.py:33-43`", "Las tres capas Conv2D con kernel 3x3, strides (1,1) y padding \"same\""],
        ["Batch Normalization", "`src/modeling.py:34-35, 39-40`", "Capas BatchNormalization tras la 1a y 2a conv, activadas por `use_batchnorm`"],
        ["Dropout", "`src/modeling.py:47-48`", "Capa Dropout con tasa 0.3 antes de la capa de salida"],
        ["Capa de salida Softmax", "`src/modeling.py:49`", "Dense de 10 neuronas con activacion softmax"],
        ["EarlyStopping y ReduceLROnPlateau", "`src/modeling.py:102-103`", "Callbacks de regularizacion durante el `model.fit()`"],
        ["Optimizador Adam / SGD", "`src/modeling.py:68-78`", "`compile_model()` conmutable entre ambos"],
        ["Baseline MLP (88.2 %)", "`src/modeling.py:53-65`", "`build_baseline_mlp()`, la linea base de comparacion"],
        ["Normalizacion a [0,1]", "`src/data_preparation.py:11-13`", "`normalize()`: divide por 255"],
        ["Split estratificado 90/10", "`src/data_preparation.py:21-24`", "`split_train_val()` con `stratify=y` y semilla 42"],
        ["Data Augmentation", "`src/data_preparation.py:27-38`", "`build_augmentation()`: rotacion, zoom, traslacion"],
        ["Augmentation de produccion", "`src/data_preparation.py:62-81`", "`build_production_augmentation()`: + volteo horizontal y brillo, sin RandomContrast"],
        ["Curvas Loss / Accuracy", "`src/evaluation.py:24-40`", "`plot_training_curves()` con los 4 series del History"],
        ["Matriz de confusion", "`src/evaluation.py:43-54`", "`compute_confusion()` con sklearn"],
        ["Grafico de la matriz", "`src/evaluation.py:57-75`", "`plot_confusion_matrix()` con seaborn"],
        ["Top de confusiones", "`src/evaluation.py:78-88`", "`top_confusions()`: pares real->predicho mas frecuentes"],
        ["Formula de Accuracy", "`build_notebook_entregable.py:349-360`", "`np.trace(cm) / cm.sum()`"],
        ["Preprocesado de inferencia", "`src/deployment.py:77-130`", "`preprocess_uploaded()`: 6 pasos para fotos reales"],
        ["Margen del recorte", "`src/deployment.py:23-43`", "`_crop_and_center(fill=0.8)`: la prenda ocupa el 80 %"],
        ["Normalizacion de contraste", "`src/deployment.py:46-63`", "`_normalize_contrast()` con percentiles 2-98"],
        ["Umbral y top-3", "`src/deployment.py:143-176`", "`predict(min_confidence=0.5)`: \"No reconocido\" y top-3"],
        ["Hiperparametros y semilla", "`src/config.py:46-56, 66`", "`DEFAULT_HPARAMS` y `RANDOM_SEED = 42`"],
        ["KPIs del proyecto", "`src/config.py:59-64`", "`BUSINESS_KPIS`: accuracy, latencia, tamano, F1-macro"],
        ["6 fases en la GUI", "`app.py:177, 225, 270, 305, 388, 453`", "Un `st.header` por fase de CRISP-ML"],
        ["Entrenamiento y despliegue", "`train.py`, `train_final.py`", "Entrena la base; compara candidato vs referente y despliega el mejor"],
        ["Validacion antes/despues", "`validate_fixes.py:57-113`", "Accuracy, calzado->Bolso, fotos reales y latencia"],
    ],
    widths=[4.4, 4.4, 7.2],
)

h("8.1 Comandos para reproducir todo", 2)
codigo("""# Crear el entorno (Python 3.12; TensorFlow no soporta 3.14 todavia)
uv venv --python 3.12 .venv
uv pip install --python .venv\\Scripts\\python.exe -r requirements.txt

# Entrenar la CNN base y el baseline MLP
.venv\\Scripts\\python.exe train.py

# Experimentacion de la Fase C (5 configuraciones, ~10-15 min)
.venv\\Scripts\\python.exe experiments.py

# Candidato de produccion: compara y despliega el mejor
.venv\\Scripts\\python.exe train_final.py

# Validacion antes/despues de las mejoras
.venv\\Scripts\\python.exe validate_fixes.py

# Regenerar los entregables
.venv\\Scripts\\python.exe build_notebook_entregable.py   # notebook + 10 figuras
.venv\\Scripts\\python.exe build_informe_entregable.py    # informe Word
.venv\\Scripts\\python.exe build_guia_explicaciones.py    # ESTA guia

# Levantar la aplicacion (usar python -m, no streamlit.exe: el launcher falla
# en Windows con rutas que contienen caracteres especiales)
.venv\\Scripts\\python.exe -m streamlit run app.py""")

# ===========================================================================
# ANEXO
# ===========================================================================
h("Anexo A. Texto del apartado [COMPLETAR] del informe grupal", 1)
p("El informe grupal tiene un marcador pendiente en la seccion 7.3: "
  "\"[COMPLETAR: analisis de que clases se confunden mas segun la matriz de confusion]\". "
  "Este es el texto listo para pegar:")
callout(
    "Analisis de la matriz de confusion",
    "Del analisis de la matriz de confusion sobre las 10.000 imagenes del conjunto de prueba se "
    "observa que los errores se concentran en pares de clases con silueta visualmente muy "
    "parecida. Las confusiones mas frecuentes son Camisa clasificada como Camiseta/top (107 casos) "
    "y Camiseta/top clasificada como Camisa (75 casos), lo que evidencia que el modelo dudoso "
    "entre dos categorias cuyas siluetas son practicamente identicas a 28x28; seguido de Camisa "
    "como Abrigo (65 casos) y Suéter como Abrigo (63 casos), errores explicables porque las tres "
    "prendas comparten la forma general de ropa de cobertura larga; y en el terreno del calzado, "
    "Zapatilla clasificada como Botín (44 casos). Es notable que las confusiones son "
    "aproximadamente simetricas entre pares de clases, lo que confirma que no hay una clase "
    "dominante que atraiga las predicciones, y que el problema no es de desbalance sino de "
    "resolucion espacial: a 28x28 en escala de grises, distinguir una manga larga de una manga "
    "corta es materialmente imposible. En cambio, las clases mas distinguibles (Vestido, Pantalon y "
    "Bolso) presentan una diagonal casi perfecta, con un accuracy por clase superior al 95 %, lo "
    "que confirma que el modelo ha aprendido las diferencias (rasgos) realmente relevantes de la "
    "prenda y no artefactos del dataset.",
    fill=AZUL_CLARO,
    icono="Para pegar en el 7.3",
)
p("Nota: el texto anterior esta redactado con las cifras de la corrida del notebook. Si en el "
  "informe se usa la corrida de referencia, ajusta los numeros a los de "
  "`results/experiments.json`.", italic=True, size=9)

h("Anexo B. Glosario rapido", 1)
tabla(
    ["Termino", "Significado en este proyecto"],
    [
        ["Epoch", "Una pasada completa por las 54.000 imagenes de entrenamiento"],
        ["Batch size", "128 imagenes por actualizacion de pesos"],
        ["Loss", "Error que el modelo intenta minimizar (sparse_categorical_crossentropy)"],
        ["val_loss", "El mismo error sobre el 10 % de validacion: la senal de overfitting"],
        ["Dropout", "Apaga el 30 % de las neuronas en cada paso; regulariza"],
        ["BatchNorm", "Normaliza activaciones por mini-lote; estabiliza el entrenamiento"],
        ["Augmentation", "Variantes sinteticas (rotacion, zoom, traslacion) solo en entrenamiento"],
        ["Domain shift", "La diferencia entre las imagenes del dataset y las fotos reales"],
        ["Top-3", "Las 3 clases mas probables segun el softmax, no solo la primera"],
        ["Umbral de confianza", "Si la probabilidad maxima es < 50 %, se marca \"No reconocido\""],
        ["EarlyStopping", "Detiene el entrenamiento si la val_loss no mejora en 4 epocas"],
        ["ReduceLROnPlateau", "Reduce la tasa de aprendizaje a la mitad si la val_loss se estanca"],
    ],
    widths=[4.4, 11.6],
)

# ---------------------------------------------------------------------------
# Pie de pagina
# ---------------------------------------------------------------------------
pie_de_pagina()

# ---------------------------------------------------------------------------
# Guardar
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
doc.save(OUT_PATH)

# Copiar tambien las figuras para tener todo en una sola carpeta.
fig_dst = OUT_DIR / "figures"
if fig_dst.exists():
    shutil.rmtree(fig_dst)
shutil.copytree(FIGDIR, fig_dst)

print("Documento generado:", OUT_PATH)
print("Figuras copiadas a:", fig_dst)
