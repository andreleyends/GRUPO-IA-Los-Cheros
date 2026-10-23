# -*- coding: utf-8 -*-
"""5.1 Consolidado de metricas del Sprint 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import utils


def leer(nombre):
    return json.loads((config.METRICAS / nombre).read_text(encoding="utf-8"))


def main():
    d1 = leer("3_1_resumen.json")
    d2 = leer("3_2_resumen.json")
    d3 = pd.read_csv(config.METRICAS / "3_3_particiones.csv")
    m1 = leer("4_1_cnn_metricas.json")
    m2 = leer("4_2_compat_metricas.json")
    m3 = leer("4_3_taste_metricas.json")
    rank = m3["ranking"]

    filas = [
        ("Datos", "Imagenes del subconjunto", f"{d1['imagenes_finales']:,}", ""),
        ("Datos", "Instancias", f"{d1['instancias_finales']:,}", ""),
        ("Datos", "Instancias principales", f"{d1['instancias_principales']:,}", ""),
        ("Datos", "Partes", f"{d1['instancias_partes']:,}", ""),
        ("Datos", "Instancias con atributos", f"{d1['con_atributos']:,}", f"{d1['pct_con_atributos']:.1f}%"),
        ("Pares", "Positivos (incluye sinteticos)", f"{d2['positivos']:,}", ""),
        ("Pares", "Negativos", f"{d2['negativos']:,}", ""),
        ("Pares", "Sinteticos de apoyo", f"{d2['sinteticos_apoyo']:,}", ""),
        ("CNN", "Accuracy top-1", f"{m1['accuracy']:.3f}", ""),
        ("CNN", "Accuracy top-3", f"{m1['top3']:.3f}", ""),
        ("CNN", "Macro F1", f"{m1['macro_f1']:.3f}", ""),
        ("CNN", "Weighted F1", f"{m1['weighted_f1']:.3f}", ""),
        ("CNN", "Silueta (accuracy)", f"{m1['acc_silueta']:.3f}", ""),
        ("CNN", "Patron (accuracy)", f"{m1['acc_patron']:.3f}", ""),
        ("CNN", "Largo (accuracy)", f"{m1['acc_largo']:.3f}", ""),
        ("CNN", "Detalle (accuracy)", f"{m1['acc_detalle']:.3f}", ""),
        ("Compatibilidad", "AUC test", f"{m2['auc']:.3f}", ""),
        ("Compatibilidad", "AP test", f"{m2['ap']:.3f}", ""),
        ("Compatibilidad", "Accuracy test", f"{m2['accuracy']:.3f}", ""),
        ("Compatibilidad", "F1 test", f"{m2['f1']:.3f}", ""),
        ("Taste", "AUC preferencias test", f"{m3['auc']:.3f}", "sintetico"),
        ("Taste", "NDCG@10", f"{rank['taste']['ndcg10']:.3f}", "30 perfiles"),
        ("Taste + reglas", "NDCG@10", f"{rank['taste_reglas']['ndcg10']:.3f}", "30 perfiles"),
        ("Reglas", "Violacion top-10 sin reglas", f"{rank['violacion_reglas_top10_taste']:.1%}", ""),
        ("Reglas", "Violacion top-10 con reglas", f"{rank['violacion_reglas_top10_final']:.1%}", ""),
    ]
    tabla = pd.DataFrame(filas, columns=["Bloque", "Metrica", "Valor", "Nota"])
    tabla.to_csv(config.METRICAS / "5_1_resumen.csv", index=False, encoding="utf-8")
    resumen = {
        "dataset": d1, "pares": d2,
        "particiones": d3.to_dict(orient="records"),
        "cnn": m1, "compat": m2,
        "taste": {k: v for k, v in m3.items() if k != "historial"},
        "tabla": filas,
    }
    utils.json_dump(resumen, config.METRICAS / "5_1_resumen.json")

    plt = utils.configurar_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    nombres = ["CNN top-1", "CNN top-3", "Compat. AUC", "Taste NDCG@10"]
    vals = [m1["accuracy"], m1["top3"], m2["auc"], rank["taste_reglas"]["ndcg10"]]
    colores = ["#6b8f71", "#2f6f8f", "#a24b3a", "#8a6f9e"]
    ax.bar(nombres, vals, color=colores)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Valor")
    ax.set_title("Resumen de metricas por etapa")
    utils.guardar_figura(fig, "5_1_resumen.png")

    lineas = [
        "# 5.1 Consolidado de metricas",
        "",
        "Este documento reúne las metricas de las etapas 3.1 a 4.3. No son "
        "directamente comparables entre si: cada una mide una tarea distinta.",
        "",
        "## Tabla resumen",
        "| Bloque | Metrica | Valor | Nota |",
        "|---|---|---:|---|",
    ]
    for r in filas:
        lineas.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    lineas += [
        "",
        "## Lectura por etapa",
        "",
        "### Datos y pares",
        f"El subconjunto conserva **{d1['imagenes_finales']:,} imagenes** y "
        f"**{d1['instancias_finales']:,} instancias**. La cobertura de atributos es "
        f"**{d1['pct_con_atributos']:.1f}%**. Se generaron **{d2['positivos']:,} "
        f"positivos** y **{d2['negativos']:,} negativos**, con "
        f"**{d2['sinteticos_apoyo']}** pares sinteticos de apoyo.",
        "",
        "### CNN",
        f"La CNN obtiene **{m1['accuracy']:.3f}** de accuracy top-1 y "
        f"**{m1['top3']:.3f}** top-3 en test. El macro F1 es "
        f"**{m1['macro_f1']:.3f}**, afectado por clases con muy pocos ejemplos. "
        f"La cabeza de patron es la mas fuerte ({m1['acc_patron']:.3f}) y la de "
        f"silueta la mas debil ({m1['acc_silueta']:.3f}).",
        "",
        "### Compatibilidad",
        f"El MLP de compatibilidad alcanza AUC **{m2['auc']:.3f}**, AP "
        f"**{m2['ap']:.3f}** y F1 **{m2['f1']:.3f}**. Los negativos aleatorios "
        "tienen score medio bajo y los pares positivos score medio alto; los "
        "negativos dificiles quedan cerca del umbral, como se esperaba.",
        "",
        "### Personalizacion",
        f"El MLP de taste obtiene AUC **{m3['auc']:.3f}** en preferencias sinteticas. "
        f"El NDCG@10 es **{rank['taste']['ndcg10']:.3f}** y baja levemente a "
        f"**{rank['taste_reglas']['ndcg10']:.3f}** al aplicar reglas. La afinidad "
        f"oraculo obtiene **{rank['afinidad']['ndcg10']:.3f}**, por lo que el modelo "
        "aprende el orden pero no supera la funcion que genero las etiquetas.",
        "",
        "### Reglas",
        f"Las reglas reducen la violacion en el top-10 de "
        f"**{rank['violacion_reglas_top10_taste']:.1%}** a "
        f"**{rank['violacion_reglas_top10_final']:.1%}**, con una perdida pequena de "
        "NDCG. Es el intercambio esperado entre personalizacion y coherencia.",
        "",
        "## Evidencia visual",
        "![Resumen](figuras/5_1_resumen.png)",
        "",
        "![CNN](figuras/4_1_curvas.png)",
        "",
        "![ROC compatibilidad](figuras/4_2_roc.png)",
        "",
        "![NDCG taste](figuras/4_3_ndcg.png)",
        "",
        "## Limitaciones",
        "- El dataset es un subconjunto de 4.000 imagenes, no el dataset completo.",
        "- Las preferencias de taste son sinteticas y no sustituyen usuarios reales.",
        "- El macro F1 de la CNN esta penalizado por clases raras; el weighted F1 "
        "resume mejor el rendimiento global.",
        "- Las metricas de compatibilidad y taste se evaluan sobre pares generados "
        "por coocurrencia y reglas, no sobre juicios humanos independientes.",
    ]
    (config.REPORTS / "5.1_Reporte_de_métricas.md").write_text("\n".join(lineas), encoding="utf-8")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
