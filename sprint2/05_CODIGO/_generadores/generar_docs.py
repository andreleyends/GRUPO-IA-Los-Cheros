# -*- coding: utf-8 -*-
"""Genera los DOCX del Sprint 2 a partir de los Markdown del proyecto."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "sprint1" / "_generadores"))

import comun  # noqa: E402

DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
SCRUM = ROOT / "scrum"

FUENTES = [
    SCRUM / "S2.0_Sprint_Goal_Y_Sprint_Backlog.md",
    REPORTS / "3.1_Dataset_limpio.md",
    REPORTS / "3.2_Dataset_de_pares_etiquetados.md",
    REPORTS / "3.3_Conjuntos_de_entrenamiento_validación_y_prueba.md",
    REPORTS / "4.1_Modelo_CNN_entrenado.md",
    REPORTS / "4.2_Modelo_MLP_de_compatibilidad.md",
    REPORTS / "4.3_MLP_de_personalización_y_filtros_de_reglas.md",
    REPORTS / "5.1_Reporte_de_métricas.md",
    REPORTS / "5.2_Reporte_comparativo_y_de_ablación.md",
    SCRUM / "S2.1_Registro_breve_de_dailies.md",
    SCRUM / "S2.2_Burndown_y_Backlog_actualizados.md",
    SCRUM / "S2.3_Acta_de_Review.md",
    SCRUM / "S2.4_Acta_de_Retrospective.md",
]


def limpiar(texto: str) -> str:
    texto = re.sub(r"\*\*(.+?)\*\*", r"\1", texto)
    texto = re.sub(r"`(.+?)`", r"\1", texto)
    return texto.strip()


def burndown():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    dias = ["20/10", "21/10", "22/10", "23/10"]
    puntos = [42, 35, 22, 0]
    ax.plot(dias, puntos, marker="o", color="#1f3b63", linewidth=2)
    for x, y in zip(dias, puntos):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_ylim(0, 45)
    ax.set_ylabel("Puntos restantes")
    ax.set_title("Burndown planificado - Sprint 2")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "figuras" / "S2_2_burndown.png", dpi=180)
    plt.close(fig)


def convertir(md: Path):
    doc = comun.nuevo_documento(md.stem)
    doc.core_properties.author = "Equipo StyleMate - Sprint 2"
    lineas = md.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lineas):
        linea = lineas[i].rstrip()
        if not linea.strip():
            i += 1
            continue
        if linea.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", linea)
            if m:
                alt, rel = m.group(1), m.group(2)
                ruta = (md.parent / rel).resolve()
                if ruta.exists():
                    comun.figura(doc, ruta, limpiar(alt),
                                 ancho_cm=15.5 if "5_2" not in md.stem else 14.0)
            i += 1
            continue
        if linea.startswith("|"):
            bloque = []
            while i < len(lineas) and lineas[i].strip().startswith("|"):
                bloque.append(lineas[i].strip())
                i += 1
            if len(bloque) >= 2:
                cab = [limpiar(c) for c in bloque[0].strip("|").split("|")]
                filas = []
                for b in bloque[2:]:
                    filas.append([limpiar(c) for c in b.strip("|").split("|")])
                comun.tabla(doc, cab, filas, size=9 if len(cab) <= 4 else 8)
            continue
        if linea.startswith("### "):
            comun.h(doc, limpiar(linea[4:]), 3)
        elif linea.startswith("## "):
            comun.h(doc, limpiar(linea[3:]), 2)
        elif linea.startswith("# "):
            comun.h(doc, limpiar(linea[2:]), 1)
        elif linea.startswith("- "):
            items = []
            while i < len(lineas) and lineas[i].strip().startswith("- "):
                items.append(limpiar(lineas[i].strip()[2:]))
                i += 1
            comun.vinetas(doc, items)
            continue
        elif re.match(r"^\d+\.\s+", linea):
            items = []
            while i < len(lineas) and re.match(r"^\d+\.\s+", lineas[i].strip()):
                items.append(limpiar(re.sub(r"^\d+\.\s+", "", lineas[i].strip())))
                i += 1
            comun.numeradas(doc, items)
            continue
        elif linea.strip() == "---":
            pass
        else:
            comun.p(doc, limpiar(linea))
        i += 1
    salida = DOCS / f"{md.stem}.docx"
    comun.guardar(doc, salida)
    print(f"{salida.relative_to(ROOT)}")


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    burndown()
    for md in FUENTES:
        if not md.exists():
            print(f"FALTA: {md}")
            continue
        convertir(md)


if __name__ == "__main__":
    main()
