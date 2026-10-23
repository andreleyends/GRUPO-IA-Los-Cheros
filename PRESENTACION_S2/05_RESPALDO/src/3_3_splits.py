# -*- coding: utf-8 -*-
"""
3.3 Particiones train/val/test y aumento de datos.

La particion se hace por imagen (nunca por instancia) y se estratifica por
categoria dominante. Los pares de 3.2 se generan despues dentro de cada split.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import pares, utils

utils.ensure_dirs()


def split_estratificado(meta: pd.DataFrame, col: str = "dominant_main",
                        seed: int = config.SEED) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    mapa: dict[int, str] = {}
    for clase, grupo in meta.groupby(col):
        idx = grupo["id"].astype(int).to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        if n < 5:
            for i in idx:
                mapa[int(i)] = "train"
            continue
        n_val = max(1, int(round(0.10 * n)))
        n_test = max(1, int(round(0.10 * n)))
        if n_val + n_test >= n:
            n_val = max(1, (n - 1) // 2)
            n_test = max(1, n - 1 - n_val)
        if n_val + n_test >= n:
            n_val, n_test = 0, 0
        for i in idx[:n_test]:
            mapa[int(i)] = "test"
        for i in idx[n_test:n_test + n_val]:
            mapa[int(i)] = "val"
        for i in idx[n_test + n_val:]:
            mapa[int(i)] = "train"
    return mapa


def main():
    t0 = time.time()
    utils.set_seed(config.SEED)
    meta = pd.read_csv(config.PROCESSED / "imagenes_subset.csv")
    inst = pd.read_csv(config.PROCESSED / "instancias_limpias.csv")
    mapa = split_estratificado(meta)
    meta["split"] = meta["id"].astype(int).map(mapa)
    inst["split"] = inst["image_id"].astype(int).map(mapa)
    inst = inst[inst["split"].notna()].copy()

    meta.to_csv(config.PROCESSED / "imagenes_subset.csv", index=False, encoding="utf-8")
    inst.to_csv(config.PROCESSED / "instancias_split.csv", index=False, encoding="utf-8")
    for split in ["train", "val", "test"]:
        sub = meta[meta["split"] == split]
        sub.to_csv(config.PROCESSED / f"split_{split}.csv", index=False, encoding="utf-8")

    utils.json_dump({
        "fecha": utils.ahora(),
        "semilla": config.SEED,
        "criterio": "por imagen, estratificado por categoria dominante",
        "imagenes": {str(k): v for k, v in mapa.items()},
    }, config.PROCESSED / "splits.json")

    # Conteos.
    filas = []
    for split in ["train", "val", "test"]:
        sub_i = inst[inst["split"] == split]
        sub_m = meta[meta["split"] == split]
        filas.append({
            "split": split,
            "imagenes": int(len(sub_m)),
            "instancias": int(len(sub_i)),
            "principales": int(sub_i["es_principal"].sum()),
            "partes": int((~sub_i["es_principal"]).sum()),
            "con_atributos": int((sub_i["n_atributos"] > 0).sum()),
            "pares_positivos": int(pares.contar_positivos(sub_i, mapa)[split]),
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.METRICAS / "3_3_particiones.csv", index=False, encoding="utf-8")

    # Figuras.
    plt = utils.configurar_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(tabla["split"], tabla["imagenes"], color=["#2f6f8f", "#c9a227", "#8a6f4d"])
    ax.set_ylabel("Imagenes")
    ax.set_title("Particion por imagen (80/10/10)")
    for x, y in zip(tabla["split"], tabla["imagenes"]):
        ax.text(x, y, f"{y:,}", ha="center", va="bottom", fontsize=9)
    utils.guardar_figura(fig, "3_3_particiones.png")

    top = inst.groupby(["split", "category_name"]).size().reset_index(name="n")
    top_cats = top.groupby("category_name")["n"].sum().sort_values(ascending=False).head(15).index
    piv = top[top["category_name"].isin(top_cats)].pivot_table(
        index="category_name", columns="split", values="n", aggfunc="sum", fill_value=0)
    piv = piv.reindex(top_cats[::-1])
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    left = np.zeros(len(piv))
    for split, color in zip(["train", "val", "test"], ["#2f6f8f", "#c9a227", "#8a6f4d"]):
        vals = piv.get(split, pd.Series(0, index=piv.index)).values
        ax.barh(piv.index, vals, left=left, color=color, label=split)
        left += vals
    ax.set_xlabel("Instancias")
    ax.set_title("Distribucion de categorias por particion (top 15)")
    ax.legend(frameon=False)
    utils.guardar_figura(fig, "3_3_categorias.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(tabla["split"], tabla["pares_positivos"],
           color=["#2f6f8f", "#c9a227", "#8a6f4d"])
    ax.set_ylabel("Pares positivos")
    ax.set_title("Pares positivos disponibles por particion")
    utils.guardar_figura(fig, "3_3_pares.png")

    # Reporte.
    lineas = [
        "# 3.3 Particiones train/val/test y aumento de datos",
        "",
        "## Objetivo",
        "Asignar imagenes completas a train, val y test para evitar fuga de "
        "informacion, y definir el esquema de aumento usado por GarmentNet.",
        "",
        "## Metodologia",
        "- Particion **por imagen**, nunca por instancia.",
        "- Estratificacion por **categoria dominante** de cada imagen.",
        "- Proporcion objetivo **80/10/10**; las categorias con menos de 5 imagenes "
        "quedan completas en train para no dejar clases sin representacion.",
        "- Los pares de compatibilidad se generan despues dentro de cada split (3.2), "
        "de modo que un par negativo nunca cruza particiones.",
        "",
        "## Resultados",
        "| Split | Imagenes | Instancias | Principales | Partes | Con atributos | Pares positivos |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in filas:
        lineas.append(
            f"| {r['split']} | {r['imagenes']:,} | {r['instancias']:,} | "
            f"{r['principales']:,} | {r['partes']:,} | {r['con_atributos']:,} | "
            f"{r['pares_positivos']:,} |")
    lineas += [
        "",
        f"Total: **{int(tabla['imagenes'].sum()):,}** imagenes y "
        f"**{int(tabla['pares_positivos'].sum()):,}** pares positivos.",
        "",
        "## Aumento de datos",
        "| Tecnica | Parametro | Aplicacion |",
        "|---|---|---|",
        "| Volteo horizontal | p=0.5 | Solo train |",
        "| Afinidad geometrica | rotacion +/-8 grados, traslacion 4 %, escala 0.92-1.08 | Solo train |",
        "| Color jitter | brillo/contraste 0.15, saturacion 0.12 | Solo train |",
        "| Normalizacion | media y desviacion 0.5 | Train/val/test |",
        "",
        "Val y test se evaluan sin aumento, solo con normalizacion, para que las "
        "metricas sean comparables.",
        "",
        "## Evidencia visual",
        "![Particiones](figuras/3_3_particiones.png)",
        "",
        "![Categorias](figuras/3_3_categorias.png)",
        "",
        "![Pares](figuras/3_3_pares.png)",
        "",
        "## Archivos",
        "- `data/processed/splits.json`: mapa imagen -> split.",
        "- `data/processed/split_train.csv`, `split_val.csv`, `split_test.csv`.",
        "- `data/processed/instancias_split.csv`: instancias con split.",
        "- `reports/metricas/3_3_particiones.csv`.",
        "",
        "## Limitaciones",
        "- La estratificacion es por categoria dominante; las imagenes con varias "
        "categorias principales no se estratifican por todas ellas.",
        "- La muestra de 4.000 imagenes reduce el numero de ejemplos por split "
        "frente al dataset completo.",
        "",
        f"Tiempo de ejecucion: {round((time.time()-t0)/60, 2)} min.",
    ]
    (config.REPORTS / "3.3_Conjuntos_de_entrenamiento_validación_y_prueba.md").write_text("\n".join(lineas), encoding="utf-8")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
