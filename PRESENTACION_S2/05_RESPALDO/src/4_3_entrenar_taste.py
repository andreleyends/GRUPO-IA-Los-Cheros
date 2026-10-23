# -*- coding: utf-8 -*-
"""4.3 MLP de personalizacion + motor de reglas de gusto."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import estilo, utils

utils.ensure_dirs()

CONTEXTO = {
    "casual": {"clima": "entretiempo", "ocasion": "diario", "formalidad": "casual"},
    "formal": {"clima": "entretiempo", "ocasion": "oficina", "formalidad": "formal"},
    "deportivo": {"clima": "entretiempo", "ocasion": "deporte", "formalidad": "sport"},
    "elegante": {"clima": "entretiempo", "ocasion": "noche", "formalidad": "formal"},
    "bohemio": {"clima": "calor", "ocasion": "viaje", "formalidad": "casual"},
}

COLORES_VIVOS = {"rojo", "naranja", "amarillo", "verde", "morado", "rosa"}

TOKEN_CATEGORIA = {
    "shirt, blouse": "shirt", "top, t-shirt, sweatshirt": "top",
    "sweater": "sweater", "cardigan": "cardigan", "jacket": "jacket",
    "vest": "vest", "pants": "pants", "shorts": "shorts", "skirt": "skirt",
    "coat": "coat", "dress": "dress", "jumpsuit": "jumpsuit", "cape": "cape",
    "glasses": "glasses", "hat": "hat", "tie": "tie", "glove": "glove",
    "watch": "watch", "belt": "belt", "sock": "sock", "shoe": "shoe",
    "bag, wallet": "bag", "scarf": "scarf", "umbrella": "umbrella",
}


class TasteMLP(nn.Module):
    def __init__(self, inp=161, hidden=(128, 64)):
        super().__init__()
        capas, prev = [], inp
        for h in hidden:
            capas += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                      nn.Dropout(0.2)]
            prev = h
        capas.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*capas)

    def forward(self, x):
        return self.mlp(x).squeeze(1)


def cargar_perfiles():
    ruta = config.PROCESSED / "perfiles_sinteticos.json"
    if not ruta.exists():
        perfiles = []
        for split, n, ruido in [
            ("train", config.N_PROFILES_TRAIN, config.NOISE_TRAIN),
            ("val", config.N_PROFILES_VAL, config.NOISE_VAL),
            ("test", config.N_PROFILES_TEST, config.NOISE_TEST),
        ]:
            perfiles.extend(estilo.generar_perfiles(n, split, ruido, config.SEED + n))
        ruta.write_text(json.dumps(perfiles, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return json.loads(ruta.read_text(encoding="utf-8"))


def afinidades(perfiles, embs, estilos, pares):
    """Afinidad vectorizada perfil-par, equivalente a estilo.afinidad_perfil_par."""
    style = 0.5 * (estilos[pares["pos_a"].to_numpy()] +
                   estilos[pares["pos_b"].to_numpy()])
    p = np.vstack([np.asarray(x["vector"], dtype=np.float32) for x in perfiles])
    ps = style / np.maximum(np.linalg.norm(style, axis=1, keepdims=True), 1e-8)
    pn = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-8)
    cos = ps @ pn.T
    activas = p > 0.15
    out = np.zeros((len(pares), len(perfiles)), dtype=np.float32)
    for j in range(len(perfiles)):
        idx = np.nonzero(activas[j])[0]
        cob = (style[:, idx] > 0.05).mean(axis=1) if len(idx) else np.zeros(len(pares))
        out[:, j] = 0.75 * cos[:, j] + 0.25 * cob
    return out


def construir_preferencias(perfiles, afinidad, pares):
    filas = []
    for split in ["train", "val", "test"]:
        ids = [i for i, p in enumerate(perfiles) if p["split"] == split]
        sub = pares[pares["split"] == split].reset_index(drop=True)
        a = afinidad[pares["split"].to_numpy() == split][:, ids]
        for j, ip in enumerate(ids):
            orden = np.argsort(-a[:, j], kind="stable")
            pos = orden[:config.PREF_POS_POR_PERFIL]
            neg = orden[-config.PREF_NEG_POR_PERFIL:]
            for idx in pos:
                filas.append((perfiles[ip]["profile_id"], split,
                              perfiles[ip]["arquetipo"], sub.loc[idx, "pair_id"],
                              int(sub.loc[idx, "label"]), 1, float(a[idx, j])))
            for idx in neg:
                filas.append((perfiles[ip]["profile_id"], split,
                              perfiles[ip]["arquetipo"], sub.loc[idx, "pair_id"],
                              int(sub.loc[idx, "label"]), 0, float(a[idx, j])))
    return pd.DataFrame(filas, columns=["profile_id", "split", "arquetipo", "pair_id",
                                        "label_compat", "label", "afinidad"])


def features(perfil, style_pair, compat):
    perfil = torch.as_tensor(perfil, dtype=torch.float32)
    style_pair = torch.as_tensor(style_pair, dtype=torch.float32)
    x = torch.cat([perfil, style_pair, perfil * style_pair,
                   torch.abs(perfil - style_pair),
                   torch.tensor([compat], dtype=torch.float32)])
    return x


def matriz_features(perfiles, pares, estilos, embs):
    style = 0.5 * (estilos[pares["pos_a"].to_numpy()] +
                   estilos[pares["pos_b"].to_numpy()]).astype(np.float32)
    comp = pares["compat_score"].to_numpy(dtype=np.float32)
    # Se rellena la afinidad por perfil en el bucle externo.
    return style, comp


def entrenar(perfiles, prefs, pares, estilos):
    pord = {p["profile_id"]: p for p in perfiles}
    style_all = 0.5 * (estilos[pares["pos_a"].to_numpy()] +
                       estilos[pares["pos_b"].to_numpy()]).astype(np.float32)
    pair_idx = {int(pid): i for i, pid in enumerate(pares["pair_id"])}
    comp_all = pares["compat_score"].to_numpy(dtype=np.float32)

    def construir(split):
        sub = prefs[prefs["split"] == split]
        xs, ys = [], []
        for pid, arq, pair_id, y, af in sub[["profile_id", "arquetipo", "pair_id",
                                             "label", "afinidad"]].itertuples(index=False):
            i = pair_idx[int(pair_id)]
            x = features(pord[pid]["vector"], style_all[i], comp_all[i])
            xs.append(x)
            ys.append(float(y))
        return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)

    xt, yt = construir("train")
    xv, yv = construir("val")
    dl = DataLoader(TensorDataset(xt, yt), batch_size=512, shuffle=True)
    model = TasteMLP()
    print("TasteMLP parametros:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=config.TASTE_LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.TASTE_EPOCHS)
    hist, mejor, paciencia = [], 0.0, 0
    for epoch in range(1, config.TASTE_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in tqdm(dl, desc=f"taste {epoch}", leave=False):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).numpy()
        auc = roc_auc_score(yv.numpy(), pv)
        f1 = f1_score(yv.numpy(), (pv >= 0.5).astype(int), zero_division=0)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                     "val_auc": float(auc), "val_f1": float(f1)})
        print(f"epoch {epoch:02d} loss {hist[-1]['train_loss']:.4f} "
              f"val_auc {auc:.4f} val_f1 {f1:.4f}")
        if auc > mejor:
            mejor, paciencia = auc, 0
            torch.save({"state_dict": model.state_dict(), "inp": 161,
                        "epoch": epoch, "val_auc": float(auc)}, config.MODELS / "taste_head.pt")
        else:
            paciencia += 1
            if paciencia >= 5:
                break
    pd.DataFrame(hist).to_csv(config.METRICAS / "taste_historial.csv",
                              index=False, encoding="utf-8")
    ckpt = torch.load(config.MODELS / "taste_head.pt", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, style_all, pair_idx, comp_all, hist, ckpt["epoch"]


def predecir_prefs(model, perfiles, prefs, style_all, pair_idx, comp_all):
    pord = {p["profile_id"]: p for p in perfiles}
    xs, ys = [], []
    for pid, pair_id, y, af in prefs[["profile_id", "pair_id", "label", "afinidad"]].itertuples(index=False):
        i = pair_idx[int(pair_id)]
        xs.append(features(pord[pid]["vector"], style_all[i], comp_all[i]))
        ys.append(float(y))
    with torch.no_grad():
        p = torch.sigmoid(model(torch.stack(xs))).numpy()
    return p, np.asarray(ys)


def prendas_reglas(pair, color_a, color_b, lexico):
    tokens = []
    for nombre in [pair["nombre_a"], pair["nombre_b"]]:
        t = TOKEN_CATEGORIA.get(nombre)
        if t:
            tokens.append(t)
    style = 0.5 * (np.asarray(lexico["vec_a"], dtype=np.float32) +
                   np.asarray(lexico["vec_b"], dtype=np.float32))
    dims = estilo.DIM
    if style[dims["formalidad.sport"]] > 0.25:
        tokens.append("sport")
    for k, token in [("estampado.grafico", "estampado"), ("estampado.floral", "floral"),
                     ("estampado.rayas", "rayas"), ("estampado.cuadros", "cuadros")]:
        if style[dims[k]] > 0.25:
            tokens.append(token)
    ctx = dict(CONTEXTO.get(pair["arquetipo"], CONTEXTO["casual"]))
    ctx["colores_vivos"] = (int(color_a in COLORES_VIVOS) +
                            int(color_b in COLORES_VIVOS) +
                            int(style[dims["color_base.vivo"]] > 0.30))
    ctx["material"] = "denim" if style[dims["material.denim"]] > 0.35 else "tela"
    return tokens, ctx


def ranking_metricas(scores, relevantes, k=10):
    orden = np.argsort(-scores, kind="stable")
    top = orden[:k]
    rel = relevantes[top].astype(float)
    dcg = float(np.sum(rel / np.log2(np.arange(2, 2 + len(rel)))))
    ideal = np.zeros(k)
    ideal[: int(min(k, relevantes.sum()))] = 1.0
    idcg = float(np.sum(ideal / np.log2(np.arange(2, 2 + k))))
    hit = float(rel.sum() > 0)
    pos = np.nonzero(relevantes[orden] > 0)[0]
    mrr = float(1.0 / (pos[0] + 1)) if len(pos) and pos[0] < k else 0.0
    return dcg / idcg if idcg > 0 else 0.0, hit, mrr


def evaluar_ranking(model, perfiles, pares_test, style_all, pair_idx, comp_all,
                    estilos, colores):
    model.eval()
    res = {"compat": [], "afinidad": [], "taste": [], "taste_reglas": []}
    ejemplos = []
    viol_taste, viol_reglas = [], []
    for p in [x for x in perfiles if x["split"] == "test"]:
        idx = pares_test.index.to_numpy()
        style = style_all[idx]
        comp = comp_all[idx]
        perfil = np.asarray(p["vector"], dtype=np.float32)
        x = torch.tensor(np.concatenate([np.tile(perfil, (len(idx), 1)), style,
                                         np.tile(perfil, (len(idx), 1)) * style,
                                         np.abs(np.tile(perfil, (len(idx), 1)) - style),
                                         comp[:, None]], axis=1), dtype=torch.float32)
        with torch.no_grad():
            taste = torch.sigmoid(model(x)).numpy()
        af = afinidades([p], None, estilos, pares_test)[:, 0]
        relevantes = np.zeros(len(idx), dtype=int)
        relevantes[np.argsort(-af)[:config.PREF_POS_POR_PERFIL]] = 1

        penales = []
        for row in pares_test.itertuples(index=False):
            tokens, ctx = prendas_reglas({
                "nombre_a": row.nombre_a, "nombre_b": row.nombre_b,
                "arquetipo": p["arquetipo"],
            }, colores[int(row.pos_a)], colores[int(row.pos_b)], {
                "vec_a": estilos[int(row.pos_a)], "vec_b": estilos[int(row.pos_b)]})
            pen, _ = estilo.RULES.aplicar_reglas(tokens, ctx)
            penales.append(pen)
        penales = np.asarray(penales, dtype=np.float32)
        final = taste - 0.5 * penales

        for nombre, sc in [("compat", comp), ("afinidad", af),
                           ("taste", taste), ("taste_reglas", final)]:
            res[nombre].append(ranking_metricas(sc, relevantes))
        viol_taste.append(float(np.mean(penales[np.argsort(-taste)[:10]] > 0)))
        viol_reglas.append(float(np.mean(penales[np.argsort(-final)[:10]] > 0)))

        orden = np.argsort(-final)[:5]
        for rank, i in enumerate(orden, 1):
            row = pares_test.iloc[i]
            ejemplos.append({
                "profile_id": p["profile_id"], "arquetipo": p["arquetipo"],
                "rank": rank, "pair_id": int(row["pair_id"]),
                "prendas": f"{row['nombre_a']} + {row['nombre_b']}",
                "compat": float(comp[i]), "afinidad": float(af[i]),
                "taste": float(taste[i]), "penalizacion": float(penales[i]),
                "score_final": float(final[i]),
            })
    out = {}
    for k, vals in res.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = {"ndcg10": float(arr[:, 0].mean()), "hr10": float(arr[:, 1].mean()),
                  "mrr10": float(arr[:, 2].mean())}
    out["violacion_reglas_top10_taste"] = float(np.mean(viol_taste))
    out["violacion_reglas_top10_final"] = float(np.mean(viol_reglas))
    return out, pd.DataFrame(ejemplos)


def main():
    t0 = time.time()
    utils.set_seed(config.SEED)
    perfiles = cargar_perfiles()
    pares = pd.read_csv(config.PROCESSED / "pares_scored.csv")
    estilos = np.load(config.PROCESSED / "estilos_instancias.npy")
    inst = pd.read_csv(config.PROCESSED / "instancias_estilo.csv").sort_values("pos")
    colores = inst["color_nombre"].fillna("").to_numpy()
    embs = np.load(config.PROCESSED / "embeddings.npy")

    # Afinidad completa y preferencias sinteticas.
    afinidad = afinidades(perfiles, embs, estilos, pares)
    prefs = construir_preferencias(perfiles, afinidad, pares)
    prefs.to_csv(config.PROCESSED / "preferencias_sinteticas.csv", index=False,
                 encoding="utf-8")

    model, style_all, pair_idx, comp_all, hist, mejor_epoch = entrenar(
        perfiles, prefs, pares, estilos)
    test_prefs = prefs[prefs["split"] == "test"]
    p_test, y_test = predecir_prefs(model, perfiles, test_prefs, style_all, pair_idx, comp_all)
    mt = {
        "auc": float(roc_auc_score(y_test, p_test)),
        "ap": float(average_precision_score(y_test, p_test)),
        "accuracy": float(accuracy_score(y_test, (p_test >= 0.5).astype(int))),
        "f1": float(f1_score(y_test, (p_test >= 0.5).astype(int), zero_division=0)),
        "n": int(len(y_test)), "mejor_epoch": int(mejor_epoch),
        "n_perfiles": len([p for p in perfiles if p["split"] == "test"]),
    }

    pares_test = pares[pares["split"] == "test"]
    rank, ejemplos = evaluar_ranking(model, perfiles, pares_test, style_all,
                                     pair_idx, comp_all, estilos, colores)
    mt["ranking"] = rank
    mt["duracion_min"] = round((time.time() - t0) / 60, 2)
    mt["historial"] = hist
    utils.json_dump(mt, config.METRICAS / "4_3_taste_metricas.json")
    ejemplos.to_csv(config.PROCESSED / "recomendaciones_test.csv", index=False,
                    encoding="utf-8")

    # Figuras.
    plt = utils.configurar_matplotlib()
    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist],
             color="#a24b3a", label="pérdida train")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("BCE", color="#a24b3a")
    ax2 = ax1.twinx()
    ax2.plot([h["epoch"] for h in hist], [h["val_auc"] for h in hist],
             color="#2f6f8f", label="AUC val")
    ax2.set_ylabel("AUC val", color="#2f6f8f")
    ax1.set_title("Entrenamiento del MLP de personalización")
    utils.guardar_figura(fig, "4_3_curvas.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    nombres = ["Compatibilidad", "Afinidad (oráculo)", "Taste", "Taste + reglas"]
    vals = [rank["compat"]["ndcg10"], rank["afinidad"]["ndcg10"],
            rank["taste"]["ndcg10"], rank["taste_reglas"]["ndcg10"]]
    ax.bar(nombres, vals, color=["#9aa7b0", "#6b8f71", "#2f6f8f", "#a24b3a"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("NDCG@10")
    ax.set_title("Ranking con perfiles de test")
    utils.guardar_figura(fig, "4_3_ndcg.png")

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(["Taste", "Taste + reglas"],
           [rank["violacion_reglas_top10_taste"], rank["violacion_reglas_top10_final"]],
           color=["#6b8f71", "#a24b3a"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Tasa de violación")
    ax.set_title("Violaciones de reglas en el top 10")
    utils.guardar_figura(fig, "4_3_reglas.png")

    lineas = [
        "# 4.3 MLP de personalización y reglas de gusto",
        "",
        "## Objetivo",
        "Aprender un ranking personalizado a partir de perfiles sintéticos y aplicar "
        "reglas duras de sentido común para evitar recomendaciones incoherentes.",
        "",
        "## Diseño",
        "- 120 perfiles train, 30 val y 30 test, disjuntos, con 5 arquetipos y ruido "
        "creciente (0,10 / 0,15 / 0,20).",
        "- Cada perfil genera 250 pares preferidos y 250 no preferidos por afinidad "
        "de estilo; los pares conservan su etiqueta de compatibilidad.",
        "- Entrada (161): perfil (40), estilo medio del par (40), interacción (40), "
        "diferencia absoluta (40) y compatibilidad CNN (1). La afinidad no se usa "
        "como entrada para evitar fuga directa desde la etiqueta sintética.",
        "- MLP 161 -> 128 -> 64 -> 1, BCE, AdamW y early stopping.",
        "- Reglas de Sprint 1: shorts con frío, deportivo en oficina, denim formal, "
        "doble estampado y exceso de colores vivos.",
        "",
        "## Clasificación de preferencias (test)",
        f"- AUC: **{mt['auc']:.3f}**; AP: **{mt['ap']:.3f}**.",
        f"- Accuracy: **{mt['accuracy']:.3f}**; F1: **{mt['f1']:.3f}**.",
        f"- Pares evaluados: **{mt['n']:,}**; perfiles: **{mt['n_perfiles']}**.",
        f"- Mejor época: **{mt['mejor_epoch']}**.",
        "",
        "## Ranking (30 perfiles test, todos los pares test)",
        "| Sistema | NDCG@10 | HR@10 | MRR@10 |",
        "|---|---:|---:|---:|",
    ]
    for k, etiqueta in [("compat", "Compatibilidad"), ("afinidad", "Afinidad (oráculo)"),
                        ("taste", "Taste"), ("taste_reglas", "Taste + reglas")]:
        m = rank[k]
        lineas.append(f"| {etiqueta} | {m['ndcg10']:.3f} | {m['hr10']:.3f} | "
                      f"{m['mrr10']:.3f} |")
    lineas += [
        "",
        f"- Violación de reglas en top-10: **{rank['violacion_reglas_top10_taste']:.1%}** "
        f"con Taste y **{rank['violacion_reglas_top10_final']:.1%}** con Taste + reglas.",
        "",
        "## Ejemplos de recomendación",
        "| Perfil | Arquetipo | Rank | Prendas | Compat. | Taste | Penal. | Final |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in ejemplos.head(15).itertuples(index=False):
        lineas.append(f"| {r.profile_id} | {r.arquetipo} | {r.rank} | {r.prendas} | "
                      f"{r.compat:.3f} | {r.taste:.3f} | {r.penalizacion:.2f} | "
                      f"{r.score_final:.3f} |")
    lineas += [
        "",
        "## Evidencia visual",
        "![Curvas](figuras/4_3_curvas.png)",
        "",
        "![NDCG](figuras/4_3_ndcg.png)",
        "",
        "![Reglas](figuras/4_3_reglas.png)",
        "",
        "## Archivos",
        "- `models/taste_head.pt`: checkpoint del MLP.",
        "- `data/processed/perfiles_sinteticos.json`: perfiles y vectores 40D.",
        "- `data/processed/preferencias_sinteticas.csv`: pares elegidos por perfil.",
        "- `data/processed/recomendaciones_test.csv`: top-5 por perfil de test.",
        "- `reports/metricas/4_3_taste_metricas.json`.",
        "",
        "## Limitaciones",
        "- Las preferencias son sintéticas: la afinidad no sustituye juicios humanos.",
        "- El ranking se evalúa contra la misma función de afinidad que genera las "
        "etiquetas; mide consistencia con el diseño, no validez de mercado.",
        "- El score de compatibilidad de train es in-sample; en val/test es out-of-sample.",
        "",
        f"Tiempo de ejecución: {mt['duracion_min']:.1f} min.",
    ]
    (config.REPORTS / "4.3_MLP_de_personalización_y_filtros_de_reglas.md").write_text("\n".join(lineas), encoding="utf-8")
    print(json.dumps({k: v for k, v in mt.items() if k != "historial"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
