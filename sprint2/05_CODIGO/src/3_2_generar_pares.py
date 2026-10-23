# -*- coding: utf-8 -*-
"""
3.2 Generacion de pares y etiquetas de compatibilidad (seccion 2.5).

Positivos: categorias principales distintas que coocurren en la misma imagen.
Negativos: instancias de imagenes distintas; 30 % dificiles (mismo par de
categorias que un positivo). Apoyo sintetico solo para pares con pocos
positivos. Todos los pares llevan marca de origen.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import estilo, pares, utils
from src.etiquetas import parse_attribute_ids

utils.ensure_dirs()


def preparar_estilos(inst: pd.DataFrame):
    attr = pd.read_csv(config.IDX / "atributos.csv")
    id2name = dict(zip(attr["id"].astype(int), attr["name"].astype(str)))
    inst = inst.sort_values("pos").reset_index(drop=True)
    attrs = inst["attribute_ids"].apply(parse_attribute_ids)
    estilos = []
    for i, row in enumerate(inst.itertuples()):
        nombres = [id2name.get(int(a), "") for a in attrs.iloc[i]]
        estilos.append(estilo.vector_prenda(row.category_name, nombres,
                                             str(row.color_nombre)))
    arr = np.vstack(estilos).astype(np.float32)
    np.save(config.PROCESSED / "estilos_instancias.npy", arr)
    inst[["pos", "annotation_id", "image_id", "category_id", "category_name",
          "split", "color_nombre"]].to_csv(
        config.PROCESSED / "instancias_estilo.csv", index=False, encoding="utf-8")
    return inst, arr


def _record(pair_id, split, label, origen, a, b, afinidad):
    return {
        "pair_id": int(pair_id),
        "split": split,
        "label": int(label),
        "origen": origen,
        "inst_a": int(a.annotation_id),
        "inst_b": int(b.annotation_id),
        "pos_a": int(a.pos),
        "pos_b": int(b.pos),
        "image_a": int(a.image_id),
        "image_b": int(b.image_id),
        "cat_a": int(a.category_id),
        "cat_b": int(b.category_id),
        "nombre_a": str(a.category_name),
        "nombre_b": str(b.category_name),
        "afinidad_estilo": round(float(afinidad), 5),
        "es_sintetico": int(origen == "sintetico_apoyo"),
    }


def generar_positivos(inst: pd.DataFrame, estilos: np.ndarray):
    out = []
    pid = 0
    for split in ["train", "val", "test"]:
        for image_id, sub in inst[inst["split"] == split].groupby("image_id"):
            for a, b in pares.positivos_de_imagen(sub):
                afin = estilo.coseno(estilos[int(a.pos)], estilos[int(b.pos)])
                out.append(_record(pid, split, 1, "coocurrencia", a, b, afin))
                pid += 1
    return out


def generar_negativos(inst: pd.DataFrame, estilos: np.ndarray,
                      positivos: list[dict], seed: int | None = None):
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    by_cat_split = {s: defaultdict(list) for s in ["train", "val", "test"]}
    for r in inst.itertuples():
        by_cat_split[str(r.split)][int(r.category_id)].append(r)
    pos_por_split = defaultdict(list)
    for p in positivos:
        pos_por_split[p["split"]].append(p)

    usados = {(min(p["pos_a"], p["pos_b"]), max(p["pos_a"], p["pos_b"]))
              for p in positivos}
    out = []
    pid = max([p["pair_id"] for p in positivos], default=-1) + 1

    for split in ["train", "val", "test"]:
        n_objetivo = int(round(config.NEG_RATIO * len(pos_por_split[split])))
        if n_objetivo == 0:
            continue
        pool_pos = pos_por_split[split]
        by_cat = by_cat_split[split]
        cats = [c for c, lista in by_cat.items() if len(lista) >= 2]
        for k in range(n_objetivo):
            origen = "negativo_dificil" if rng.random() < config.HARD_NEG_FRAC else "negativo_aleatorio"
            elegido = None
            for _ in range(40):
                if origen == "negativo_dificil" and pool_pos:
                    base = pool_pos[int(rng.integers(0, len(pool_pos)))]
                    ca, cb = int(base["cat_a"]), int(base["cat_b"])
                else:
                    ca, cb = int(rng.choice(cats)), int(rng.choice(cats))
                    if ca == cb:
                        continue
                if ca not in by_cat or cb not in by_cat:
                    continue
                a = by_cat[ca][int(rng.integers(0, len(by_cat[ca])))]
                b = by_cat[cb][int(rng.integers(0, len(by_cat[cb])))]
                if int(a.image_id) == int(b.image_id):
                    continue
                clave = (min(int(a.pos), int(b.pos)), max(int(a.pos), int(b.pos)))
                if clave in usados:
                    continue
                usados.add(clave)
                elegido = (a, b)
                break
            if elegido is None:
                continue
            a, b = elegido
            afin = estilo.coseno(estilos[int(a.pos)], estilos[int(b.pos)])
            out.append(_record(pid, split, 0, origen, a, b, afin))
            pid += 1
    return out


def generar_apoyo(inst: pd.DataFrame, estilos: np.ndarray,
                  positivos: list[dict], seed: int | None = None):
    rng = np.random.default_rng(config.SEED + 7 if seed is None else seed)
    train = inst[inst["split"] == "train"]
    filas = list(train.itertuples())
    by_name = defaultdict(list)
    for r in filas:
        by_name[str(r.category_name)].append(r)
    conteo = Counter()
    for p in positivos:
        if p["split"] == "train" and not p["es_sintetico"]:
            conteo[tuple(sorted([p["nombre_a"], p["nombre_b"]]))] += 1

    usados = {(min(p["pos_a"], p["pos_b"]), max(p["pos_a"], p["pos_b"]))
              for p in positivos}
    out = []
    pid = max([p["pair_id"] for p in positivos], default=-1) + 1
    for par, n in sorted(conteo.items(), key=lambda kv: kv[1]):
        if n >= config.MIN_POS_CATPAIR:
            continue
        deficit = config.MIN_POS_CATPAIR - n
        if par[0] not in by_name or par[1] not in by_name:
            continue
        candidatos = []
        for _ in range(max(80, config.MAX_SINTETICOS_POR_PAR * 4)):
            a = by_name[par[0]][int(rng.integers(0, len(by_name[par[0]])))]
            b = by_name[par[1]][int(rng.integers(0, len(by_name[par[1]])))]
            if int(a.image_id) == int(b.image_id):
                continue
            clave = (min(int(a.pos), int(b.pos)), max(int(a.pos), int(b.pos)))
            if clave in usados:
                continue
            afin = estilo.coseno(estilos[int(a.pos)], estilos[int(b.pos)])
            candidatos.append((afin, a, b, clave))
        candidatos.sort(key=lambda x: x[0], reverse=True)
        agregados = 0
        for afin, a, b, clave in candidatos:
            if agregados >= min(deficit, config.MAX_SINTETICOS_POR_PAR):
                break
            if afin < 0.15:
                continue
            usados.add(clave)
            out.append(_record(pid, "train", 1, "sintetico_apoyo", a, b, afin))
            pid += 1
            agregados += 1
    return out


def figuras(pares_df: pd.DataFrame, estilos: np.ndarray, inst: pd.DataFrame):
    plt = utils.configurar_matplotlib()

    piv = pares_df.pivot_table(index="split", columns="label", values="pair_id",
                               aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(piv.index))
    ax.bar(x - 0.2, piv.get(1, 0), width=0.4, label="Positivos", color="#2f6f8f")
    ax.bar(x + 0.2, piv.get(0, 0), width=0.4, label="Negativos", color="#a24b3a")
    ax.set_xticks(x, piv.index)
    ax.set_ylabel("Pares")
    ax.set_title("Balance de pares por particion")
    ax.legend(frameon=False)
    utils.guardar_figura(fig, "3_2_balance.png")

    orig = pares_df["origen"].value_counts()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(orig.index, orig.values, color="#4f7cac")
    ax.set_ylabel("Pares")
    ax.set_title("Pares por marca de origen")
    ax.tick_params(axis="x", rotation=20)
    utils.guardar_figura(fig, "3_2_origen.png")

    top = pares_df[pares_df["label"] == 1].copy()
    top["par"] = top.apply(lambda r: " + ".join(sorted([r["nombre_a"], r["nombre_b"]])), axis=1)
    top = top["par"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(top.index, top.values, color="#6b8f71")
    ax.set_xlabel("Pares positivos")
    ax.set_title("Top 15 combinaciones de categorias")
    utils.guardar_figura(fig, "3_2_top_pares.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for origen, color in [("coocurrencia", "#2f6f8f"), ("negativo_aleatorio", "#a24b3a"),
                          ("negativo_dificil", "#c9a227"), ("sintetico_apoyo", "#6b8f71")]:
        vals = pares_df.loc[pares_df["origen"] == origen, "afinidad_estilo"]
        if len(vals):
            ax.hist(vals, bins=30, alpha=0.55, label=origen, color=color)
    ax.set_xlabel("Afinidad de estilo (coseno)")
    ax.set_ylabel("Pares")
    ax.set_title("Distribucion de afinidad por origen")
    ax.legend(frameon=False, fontsize=8)
    utils.guardar_figura(fig, "3_2_afinidad.png")

    # Rejilla de pares positivos.
    mm = np.load(config.PROCESSED / "recortes_144.npy", mmap_mode="r")
    muestra = pares_df[(pares_df["split"] == "train") & (pares_df["label"] == 1)].head(4)
    fig, axes = plt.subplots(len(muestra), 2, figsize=(5.6, 2.5 * len(muestra)))
    if len(muestra) == 1:
        axes = np.array([axes])
    for i, (_, r) in enumerate(muestra.iterrows()):
        axes[i, 0].imshow(np.asarray(mm[int(r["pos_a"])]))
        axes[i, 0].set_title(f"{r['nombre_a'][:24]}", fontsize=8)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(np.asarray(mm[int(r["pos_b"])]))
        axes[i, 1].set_title(f"{r['nombre_b'][:24]}", fontsize=8)
        axes[i, 1].axis("off")
    utils.guardar_figura(fig, "3_2_ejemplos.png")


def main():
    t0 = time.time()
    utils.set_seed(config.SEED)
    inst = pd.read_csv(config.PROCESSED / "instancias_split.csv")
    inst, estilos = preparar_estilos(inst)
    positivos = generar_positivos(inst, estilos)
    negativos = generar_negativos(inst, estilos, positivos)
    apoyo = generar_apoyo(inst, estilos, positivos + negativos)
    pares_df = pd.DataFrame(positivos + negativos + apoyo)
    pares_df = pares_df.sort_values(["split", "pair_id"]).reset_index(drop=True)
    pares_df.to_csv(config.PROCESSED / "pares.csv", index=False, encoding="utf-8")

    resumen = {
        "fecha": utils.ahora(),
        "escenario": "B_4000_imagenes",
        "positivos": int((pares_df["label"] == 1).sum()),
        "negativos": int((pares_df["label"] == 0).sum()),
        "sinteticos_apoyo": int((pares_df["origen"] == "sintetico_apoyo").sum()),
        "ratio_neg_pos": round(float((pares_df["label"] == 0).sum() /
                                      max(1, (pares_df["label"] == 1).sum())), 3),
        "por_split": pares_df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "por_origen": pares_df["origen"].value_counts().to_dict(),
        "pares_unicos_categoria": int(pares_df[pares_df["label"] == 1]
                                      .apply(lambda r: tuple(sorted([r["nombre_a"], r["nombre_b"]])), axis=1)
                                      .nunique()),
        "duracion_min": round((time.time() - t0) / 60, 2),
    }
    utils.json_dump(resumen, config.METRICAS / "3_2_resumen.json")
    figuras(pares_df, estilos, inst)

    top = pares_df[pares_df["label"] == 1].copy()
    top["par"] = top.apply(lambda r: " + ".join(sorted([r["nombre_a"], r["nombre_b"]])), axis=1)
    top = top["par"].value_counts().head(10)

    lineas = [
        "# 3.2 Pares y etiquetas de compatibilidad",
        "",
        "## Objetivo",
        "Generar pares positivos y negativos para entrenar el MLP de compatibilidad, "
        "aplicando la estrategia 2.5 acordada con el equipo y declarando el origen "
        "de cada etiqueta.",
        "",
        "## Estrategia aplicada (2.5)",
        "- **Positivos:** instancias de categorias principales distintas que "
        "coocurren en la misma imagen. Se excluyen partes de prenda, misma categoria, "
        "imagenes con menos de dos prendas principales y combinaciones redundantes.",
        "- **Negativos:** instancias de imagenes distintas y categorias distintas. "
        "Proporcion **1:2** en esta muestra de 4.000 imagenes; el 30 % son "
        "**dificiles** (mismo par de categorias que un positivo, pero de imagenes "
        "diferentes).",
        "- **Apoyo sintetico:** solo para pares con menos de 20 positivos en train; "
        "se generan por afinidad de atributos, se marcan y se evaluan por separado.",
        "- **Reglas de estilo:** no se usan como fuente de etiquetas; solo la lista "
        "de pares redundantes excluye combinaciones imposibles del muestreo.",
        "- **Sin fuga:** todos los pares se generan dentro de la particion de su "
        "imagen (3.3).",
        "",
        "## Resultados",
        f"- Pares totales: **{len(pares_df):,}**.",
        f"- Positivos: **{resumen['positivos']:,}**; negativos: **{resumen['negativos']:,}** "
        f"(ratio {resumen['ratio_neg_pos']}:1).",
        f"- Pares sinteticos de apoyo: **{resumen['sinteticos_apoyo']:,}**.",
        f"- Combinaciones de categorias con al menos un positivo: "
        f"**{resumen['pares_unicos_categoria']:,}**.",
        "",
        "### Distribucion por particion",
        "| Split | Positivos | Negativos |",
        "|---|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        sub = pares_df[pares_df["split"] == split]
        lineas.append(f"| {split} | {int((sub['label']==1).sum()):,} | "
                      f"{int((sub['label']==0).sum()):,} |")
    lineas += [
        "",
        "### Top combinaciones positivas",
        "| Combinacion | Pares |",
        "|---|---:|",
    ]
    for par, n in top.items():
        lineas.append(f"| {par} | {n:,} |")
    lineas += [
        "",
        "## Tabla 12. Fuentes de etiquetas y sus limites",
        "| Etiqueta | Origen | Uso | Limite |",
        "|---|---|---|---|",
        "| Par positivo | Coocurrencia en una misma imagen | MLP de compatibilidad | La coocurrencia no equivale a compatibilidad de estilo |",
        "| Par negativo | Prendas de imagenes distintas | MLP de compatibilidad | Puede incluir pares que si combinarian |",
        "| Par sintetico de apoyo | Afinidad de atributos | Solo si faltan positivos | Se evalua por separado |",
        "| Preferencia | Perfiles sinteticos con ruido | MLP de personalizacion | No representa el gusto de usuarios reales |",
        "",
        "## Evidencia visual",
        "![Balance](figuras/3_2_balance.png)",
        "",
        "![Origen](figuras/3_2_origen.png)",
        "",
        "![Top pares](figuras/3_2_top_pares.png)",
        "",
        "![Afinidad](figuras/3_2_afinidad.png)",
        "",
        "![Ejemplos](figuras/3_2_ejemplos.png)",
        "",
        "## Archivos",
        "- `data/processed/pares.csv`: pares con `label`, `origen`, `split` y afinidad.",
        "- `data/processed/estilos_instancias.npy`: vector de estilo 40D por instancia.",
        "- `data/processed/instancias_estilo.csv`: indice de estilos.",
        "- `reports/metricas/3_2_resumen.json`: resumen reproducible.",
        "",
        "## Limitaciones",
        "- La coocurrencia refleja la composicion de Fashionpedia (desfiles y eventos), "
        "no una encuesta de compatibilidad.",
        "- Los negativos aleatorios pueden contener combinaciones validas; el modelo "
        "aprende a separar el sesgo del muestreo, no una verdad absoluta.",
        "- El apoyo sintetico se limita a train y se reporta aparte para no inflar "
        "las metricas de evaluacion.",
        "",
        f"Tiempo de ejecucion: {resumen['duracion_min']:.1f} min.",
    ]
    (config.REPORTS / "3.2_Dataset_de_pares_etiquetados.md").write_text("\n".join(lineas), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
