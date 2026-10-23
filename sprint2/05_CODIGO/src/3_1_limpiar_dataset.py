# -*- coding: utf-8 -*-
"""
3.1 Limpieza del dataset (escenario B: 4.000 imagenes).

- Selecciona una muestra estratificada de imagenes con y sin pares.
- Valida categorias, bboxes, area, crowding y segmentacion.
- Genera recortes 144x144 en memmap y color dominante en Lab.
- Deja trazabilidad completa en reports/metricas/3_1_resumen.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import ijson
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import estilo, utils

utils.ensure_dirs()


def cargar_tablas():
    cats = pd.read_csv(config.IDX / "categorias.csv")
    attrs = pd.read_csv(config.IDX / "atributos.csv")
    imgs = pd.read_csv(config.IDX / "imagenes.csv")
    inst = pd.read_csv(config.IDX / "instancias.csv")
    return cats, attrs, imgs, inst


def preparar_poblacion(imgs: pd.DataFrame, inst: pd.DataFrame, cats: pd.DataFrame):
    info = cats.set_index("id")[["name", "supercategory"]]
    main_ids = set(cats.loc[~cats["supercategory"].isin(
        config.SUPERCATEGORIAS_PARTE), "id"].astype(int))
    inst = inst.copy()
    inst["category_name"] = inst["category_id"].map(info["name"])
    inst["supercategory"] = inst["category_id"].map(info["supercategory"])
    inst["es_principal"] = inst["category_id"].isin(main_ids)

    main = inst[inst["es_principal"]]
    n_main = main.groupby("image_id").size().rename("n_main")
    n_dist = main.groupby("image_id")["category_id"].nunique().rename("n_main_distinct")
    g = main.groupby(["image_id", "category_id"]).size().reset_index(name="n")
    g = g.sort_values(["image_id", "n"], ascending=[True, False])
    dom = g.drop_duplicates("image_id").set_index("image_id")["category_id"]
    dom_name = dom.map(info["name"]).rename("dominant_main")

    meta = imgs.set_index("id").join(n_main).join(n_dist).join(dom_name)
    meta[["n_main", "n_main_distinct"]] = meta[["n_main", "n_main_distinct"]].fillna(0).astype(int)
    meta["dominant_main"] = meta["dominant_main"].fillna("sin_principal")
    meta["has_pair"] = meta["n_main_distinct"] >= 2
    return meta.reset_index(), inst, main_ids


def muestrear_candidatas(meta: pd.DataFrame):
    pos = meta[meta["has_pair"]].copy()
    single = meta[(meta["n_main"] == 1)].copy()
    otros = meta[(meta["n_main"] >= 1) & ~meta["has_pair"]].copy()

    n_pos = min(config.N_POSITIVAS_MUESTRA, len(pos))
    n_single = min(config.N_SINGLE_MUESTRA, len(single))
    pos_s = utils.muestrear_estratificado(pos, n_pos, "dominant_main", config.SEED)
    pos_s["fuente_muestra"] = "con_par"
    single_s = utils.muestrear_estratificado(single, n_single, "dominant_main", config.SEED + 1)
    single_s["fuente_muestra"] = "una_principal"

    cand = pd.concat([pos_s, single_s], ignore_index=True)
    if len(cand) < config.N_IMAGENES_OBJETIVO + 200:
        faltan = config.N_IMAGENES_OBJETIVO + 200 - len(cand)
        usados = set(cand["id"])
        resto = pd.concat([pos, single, otros], ignore_index=True)
        resto = resto[~resto["id"].isin(usados)]
        if len(resto):
            extra = resto.sample(n=min(faltan, len(resto)), random_state=config.SEED + 2)
            extra = extra.copy()
            extra["fuente_muestra"] = "relleno"
            cand = pd.concat([cand, extra], ignore_index=True)
    return cand.drop_duplicates("id").reset_index(drop=True)


def escanear_anotaciones(cand: pd.DataFrame, cats: pd.DataFrame, main_ids: set[int]):
    seleccionadas = set(cand["id"].astype(int))
    dims = cand.set_index("id")[["width", "height"]].to_dict("index")
    info = cats.set_index("id")[["name", "supercategory"]]
    filtros = Counter()
    registros = []
    total_leidas = 0
    ruta = config.RAW / "instances_attributes_train2020.json"
    with open(ruta, "rb") as f:
        for a in ijson.items(f, "annotations.item"):
            total_leidas += 1
            iid = int(a.get("image_id", -1))
            if iid not in seleccionadas:
                continue
            filtros["seleccionadas"] += 1
            cid = int(a.get("category_id", -1))
            if cid not in info.index:
                filtros["categoria_desconocida"] += 1
                continue
            if int(a.get("iscrowd", 0)) != 0:
                filtros["crowd"] += 1
                continue
            bbox = a.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                filtros["bbox_invalida"] += 1
                continue
            x, y, w, h = [float(v) for v in bbox]
            width = float(dims[iid]["width"])
            height = float(dims[iid]["height"])
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            w2, h2 = x1 - x0, y1 - y0
            if w2 < 4 or h2 < 4:
                filtros["bbox_invalida"] += 1
                continue
            area = max(float(a.get("area", 0) or 0), w2 * h2)
            if area < config.MIN_AREA:
                filtros["area_pequena"] += 1
                continue
            seg = a.get("segmentation", [])
            if not seg:
                filtros["sin_segmentacion_bbox"] += 1
                seg = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            attrs = [int(v) for v in a.get("attribute_ids", [])]
            rec = {
                "annotation_id": int(a.get("id", -1)),
                "image_id": iid,
                "category_id": cid,
                "category_name": str(info.loc[cid, "name"]),
                "supercategory": str(info.loc[cid, "supercategory"]),
                "es_principal": bool(cid in main_ids),
                "bbox_x": round(x0, 2),
                "bbox_y": round(y0, 2),
                "bbox_w": round(w2, 2),
                "bbox_h": round(h2, 2),
                "area": round(area, 2),
                "iscrowd": 0,
                "n_atributos": len(attrs),
                "attribute_ids": " ".join(str(v) for v in attrs),
                "segmentation_tipo": "poligonal" if isinstance(seg, list) else "rle",
                "segmentation": seg,
            }
            registros.append(rec)
    return registros, filtros, total_leidas


def seleccionar_finales(cand: pd.DataFrame, registros: list[dict]):
    df = pd.DataFrame(registros)
    validas = df.groupby("image_id").size().rename("n_validas")
    cand = cand.join(validas, on="id")
    cand["n_validas"] = cand["n_validas"].fillna(0).astype(int)
    valid = cand[cand["n_validas"] > 0].copy()
    valid["estrato"] = valid["fuente_muestra"].astype(str) + "|" + valid["dominant_main"].astype(str)
    if len(valid) > config.N_IMAGENES_OBJETIVO:
        final = utils.muestrear_estratificado(
            valid, config.N_IMAGENES_OBJETIVO, "estrato", config.SEED + 3)
    else:
        final = valid.copy()
    final = final.drop(columns=["estrato"], errors="ignore")
    return final, df


def extraer_recortes_y_color(final: pd.DataFrame, df: pd.DataFrame):
    ids = set(final["id"].astype(int))
    df = df[df["image_id"].isin(ids)].copy()
    existentes = []
    faltantes = 0
    for fn in df["image_id"].map(final.set_index("id")["file_name"]):
        p = config.IMAGENES / str(fn)
        existentes.append(p.exists())
        if not p.exists():
            faltantes += 1
    df = df[pd.Series(existentes, index=df.index)].copy()
    df = df.reset_index(drop=True)
    df["pos"] = np.arange(len(df), dtype=int)

    n = len(df)
    mm_path = config.PROCESSED / "recortes_144.npy"
    mm = np.lib.format.open_memmap(
        mm_path, mode="w+", dtype=np.uint8,
        shape=(n, config.IMG_SIZE, config.IMG_SIZE, 3),
    )
    colores, labs, hexs = [], [], []
    errores = 0
    file_map = final.set_index("id")["file_name"].to_dict()
    grupos = list(df.groupby("image_id", sort=False))
    for iid, grupo in tqdm(grupos, desc="3.1 recortes/color", unit="img"):
        ruta = config.IMAGENES / str(file_map.get(iid, ""))
        img_bgr = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if img_bgr is None:
            errores += len(grupo)
            for _ in grupo.itertuples():
                colores.append("gris")
                labs.append([0.0, 0.0, 0.0])
                hexs.append("#808080")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        for row in grupo.itertuples():
            x, y, w, h = int(row.bbox_x), int(row.bbox_y), int(row.bbox_w), int(row.bbox_h)
            crop = img_rgb[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
            if crop.size == 0:
                crop = img_rgb
            interp = cv2.INTER_AREA if max(crop.shape[:2]) > config.IMG_SIZE else cv2.INTER_LINEAR
            crop = cv2.resize(crop, (config.IMG_SIZE, config.IMG_SIZE), interpolation=interp)
            mm[int(row.pos)] = crop
            try:
                nombre, lab, hexa = estilo.extraer_color(
                    img_rgb, row.segmentation, [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h])
            except Exception:
                nombre, lab, hexa = "gris", [0.0, 0.0, 0.0], "#808080"
                errores += 1
            colores.append(nombre)
            labs.append(lab)
            hexs.append(hexa)
    mm.flush()
    del mm
    df["color_nombre"] = colores
    df["color_lab_l"] = [x[0] for x in labs]
    df["color_lab_a"] = [x[1] for x in labs]
    df["color_lab_b"] = [x[2] for x in labs]
    df["color_hex"] = hexs
    df["file_name"] = df["image_id"].map(final.set_index("id")["file_name"])
    return df, faltantes, errores


def figuras(final: pd.DataFrame, df: pd.DataFrame, resumen: dict):
    plt = utils.configurar_matplotlib()
    import matplotlib.patches as mpatches

    # Distribucion por categoria.
    top = df.groupby(["category_name", "es_principal"]).size().reset_index(name="n")
    top["tipo"] = np.where(top["es_principal"], "Principal", "Parte")
    piv = top.pivot_table(index="category_name", columns="tipo", values="n",
                          aggfunc="sum", fill_value=0)
    piv["total"] = piv.sum(axis=1)
    piv = piv.sort_values("total", ascending=True).tail(22)
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.barh(piv.index, piv.get("Principal", 0), color="#2f6f8f", label="Principal")
    ax.barh(piv.index, piv.get("Parte", 0), left=piv.get("Principal", 0),
            color="#c9a227", label="Parte")
    ax.set_xlabel("Instancias limpias")
    ax.set_title("Distribucion de instancias por categoria")
    ax.legend(frameon=False)
    utils.guardar_figura(fig, "3_1_categorias.png")

    # Cobertura de atributos.
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    counts = df["n_atributos"].clip(upper=5).value_counts().sort_index()
    ax.bar([str(i) for i in counts.index], counts.values, color="#4f7cac")
    ax.set_xlabel("Numero de atributos por instancia (5 = 5+)")
    ax.set_ylabel("Instancias")
    ax.set_title("Cobertura de atributos")
    utils.guardar_figura(fig, "3_1_atributos.png")

    # Colores principales.
    col = df[df["es_principal"]].groupby("color_nombre").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.bar(col.index, col.values, color="#8a6f4d")
    ax.set_ylabel("Instancias principales")
    ax.set_title("Color dominante extraido (paleta Lab)")
    ax.tick_params(axis="x", rotation=35)
    utils.guardar_figura(fig, "3_1_colores.png")

    # Fuentes de muestra.
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    src = final["fuente_muestra"].value_counts()
    ax.bar(src.index, src.values, color=["#2f6f8f", "#8a6f4d", "#c9a227"][:len(src)])
    ax.set_ylabel("Imagenes")
    ax.set_title("Composicion de la muestra de 4.000 imagenes")
    utils.guardar_figura(fig, "3_1_fuentes.png")

    # Rejilla de recortes.
    mm = np.load(config.PROCESSED / "recortes_144.npy", mmap_mode="r")
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(df), size=min(16, len(df)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(7.6, 7.6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(np.asarray(mm[int(df.iloc[i]["pos"])]))
        ax.set_title(f"{df.iloc[i]['category_name'][:22]}\n{df.iloc[i]['color_nombre']}",
                     fontsize=7)
        ax.axis("off")
    utils.guardar_figura(fig, "3_1_recortes.png")


def escribir_reporte(final, df, filtros, total_leidas, resumen):
    f = resumen
    lineas = [
        "# 3.1 Limpieza del dataset",
        "",
        "**Proyecto:** StyleMate · **Sprint 2** · **Escenario B (4.000 imagenes)**.",
        "",
        "## Objetivo",
        "Construir un dataset limpio y trazable para el Sprint 2, usando una muestra "
        "estratificada de 4.000 imagenes de Fashionpedia. El dataset completo se "
        "reutilizara en una entrega posterior con mayor tiempo de computo.",
        "",
        "## Metodologia",
        "1. Se identificaron las 27 categorias principales y se separaron de las "
        "partes, cierres y decoraciones.",
        "2. Se muestrearon imagenes con pares de categorias principales y una "
        "fraccion con una sola prenda principal, estratificando por categoria dominante.",
        "3. Se validaron categoria, `iscrowd`, bbox, area minima y segmentacion; "
        "las anotaciones sin segmentacion se conservaron con mascara de bbox.",
        "4. Se generaron recortes RGB de 144x144 en un memmap y se extrajo el color "
        "dominante en espacio Lab con k-means y una paleta de 12 colores.",
        "",
        "## Resultados",
        f"- Imagenes candidatas: **{f['imagenes_candidatas']:,}**.",
        f"- Imagenes finales con al menos una instancia valida: **{f['imagenes_finales']:,}**.",
        f"- Instancias finales: **{f['instancias_finales']:,}** "
        f"({f['instancias_principales']:,} principales y {f['instancias_partes']:,} partes).",
        f"- Instancias con atributos: **{f['con_atributos']:,}** "
        f"({f['pct_con_atributos']:.1f} %).",
        f"- Imagenes con 2+ prendas principales: **{f['imagenes_con_par']:,}**.",
        f"- Tiempo de preparacion: **{f['duracion_min']:.1f} min**.",
        "",
        "### Filtros aplicados",
        "| Filtro | Anotaciones |",
        "|---|---:|",
    ]
    for k, v in filtros.most_common():
        lineas.append(f"| {k} | {v:,} |")
    lineas += [
        "",
        f"Se leyeron **{total_leidas:,}** anotaciones del JSON original; "
        f"**{f['anotaciones_finales']:,}** quedaron en el dataset limpio.",
        "",
        "## Evidencia visual",
        "![Categorias](figuras/3_1_categorias.png)",
        "",
        "![Atributos](figuras/3_1_atributos.png)",
        "",
        "![Colores](figuras/3_1_colores.png)",
        "",
        "![Fuentes](figuras/3_1_fuentes.png)",
        "",
        "![Recortes](figuras/3_1_recortes.png)",
        "",
        "## Linaje y archivos",
        "- `data/processed/imagenes_subset.csv`: imagenes seleccionadas.",
        "- `data/processed/instancias_limpias.csv`: instancias validadas.",
        "- `data/processed/recortes_144.npy`: memmap de recortes.",
        "- `data/processed/recortes_meta.csv`: indice de recortes y color.",
        "- `reports/metricas/3_1_resumen.json`: resumen reproducible.",
        "",
        "## Limitaciones",
        "- La muestra de 4.000 imagenes es representativa de Fashionpedia, pero no "
        "agota la diversidad del dataset completo.",
        "- El color se estima desde la mascara de segmentacion; en prendas con "
        "varios colores se resume el cluster dominante.",
        "- La coocurrencia no equivale a compatibilidad de estilo (ver 2.5).",
        "",
    ]
    (config.REPORTS / "3.1_Dataset_limpio.md").write_text("\n".join(lineas), encoding="utf-8")


def main():
    t0 = time.time()
    utils.set_seed(config.SEED)
    cats, attrs, imgs, inst = cargar_tablas()
    meta, inst2, main_ids = preparar_poblacion(imgs, inst, cats)
    cand = muestrear_candidatas(meta)
    print(f"Candidatas: {len(cand)} | principales={len(main_ids)}")

    registros, filtros, total_leidas = escanear_anotaciones(cand, cats, main_ids)
    print(f"Anotaciones seleccionadas: {len(registros)} | filtros={dict(filtros)}")
    final, df = seleccionar_finales(cand, registros)
    print(f"Imagenes finales: {len(final)} | instancias: {len(df)}")

    df, faltantes, errores_color = extraer_recortes_y_color(final, df)
    final.to_csv(config.PROCESSED / "imagenes_subset.csv", index=False, encoding="utf-8")
    df_out = df.drop(columns=["segmentation"], errors="ignore")
    df_out.to_csv(config.PROCESSED / "instancias_limpias.csv", index=False, encoding="utf-8")
    df_out.to_csv(config.PROCESSED / "recortes_meta.csv", index=False, encoding="utf-8")

    resumen = {
        "fecha": utils.ahora(),
        "escenario": "B_4000_imagenes",
        "semilla": config.SEED,
        "imagenes_candidatas": int(len(cand)),
        "imagenes_finales": int(final["id"].nunique()),
        "instancias_finales": int(len(df)),
        "instancias_principales": int(df["es_principal"].sum()),
        "instancias_partes": int((~df["es_principal"]).sum()),
        "con_atributos": int((df["n_atributos"] > 0).sum()),
        "pct_con_atributos": float(100 * (df["n_atributos"] > 0).mean()) if len(df) else 0.0,
        "imagenes_con_par": int((final["n_main_distinct"] >= 2).sum()),
        "anotaciones_finales": int(len(df)),
        "faltantes_imagen": int(faltantes),
        "errores_color": int(errores_color),
        "filtros": dict(filtros),
        "total_leidas": int(total_leidas),
        "categorias": int(df["category_id"].nunique()),
        "duracion_min": round((time.time() - t0) / 60, 2),
    }
    utils.json_dump(resumen, config.METRICAS / "3_1_resumen.json")
    utils.json_dump({"colores": config.PALETA_COLORES},
                    config.PROCESSED / "color_paleta.json")
    figuras(final, df, resumen)
    escribir_reporte(final, df, filtros, total_leidas, resumen)
    print(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
