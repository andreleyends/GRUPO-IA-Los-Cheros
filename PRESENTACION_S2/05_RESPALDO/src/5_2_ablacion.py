# -*- coding: utf-8 -*-
"""5.2 Ablacion de componentes del modelo de compatibilidad y de las reglas."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import utils

utils.ensure_dirs()

spec = importlib.util.spec_from_file_location(
    "m42", ROOT / "src" / "4_2_entrenar_compat.py")
m42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m42)

VARIANTES = {
    "completo": {"cnn": True, "estilo": True, "categoria": True},
    "sin_cnn": {"cnn": False, "estilo": True, "categoria": True},
    "sin_estilo": {"cnn": True, "estilo": False, "categoria": True},
    "sin_categoria": {"cnn": True, "estilo": True, "categoria": False},
    "solo_categoria": {"cnn": False, "estilo": False, "categoria": True},
}


def construir(pares, embs, estilos, cfg):
    ea = embs[pares["pos_a"].to_numpy()] if cfg["cnn"] else np.zeros(
        (len(pares), config.EMBED_DIM), dtype=np.float32)
    eb = embs[pares["pos_b"].to_numpy()] if cfg["cnn"] else np.zeros(
        (len(pares), config.EMBED_DIM), dtype=np.float32)
    sa = estilos[pares["pos_a"].to_numpy()] if cfg["estilo"] else np.zeros(
        (len(pares), 40), dtype=np.float32)
    sb = estilos[pares["pos_b"].to_numpy()] if cfg["estilo"] else np.zeros(
        (len(pares), 40), dtype=np.float32)
    ca = pares["cat_a"].to_numpy() if cfg["categoria"] else np.zeros(len(pares), dtype=int)
    cb = pares["cat_b"].to_numpy() if cfg["categoria"] else np.zeros(len(pares), dtype=int)
    ds = TensorDataset(torch.tensor(ea, dtype=torch.float32),
                       torch.tensor(eb, dtype=torch.float32),
                       torch.tensor(sa, dtype=torch.float32),
                       torch.tensor(sb, dtype=torch.float32),
                       torch.tensor(ca, dtype=torch.long),
                       torch.tensor(cb, dtype=torch.long),
                       torch.tensor(pares["label"].to_numpy(), dtype=torch.float32))
    return ds


def entrenar(nombre, cfg, pares, embs, estilos, device):
    utils.set_seed(config.SEED)
    dss = {s: construir(pares[pares["split"] == s], embs, estilos, cfg)
           for s in ["train", "val", "test"]}
    dls = {s: DataLoader(dss[s], batch_size=256, shuffle=(s == "train")) for s in dss}
    model = m42.CompatMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    mejor, paciencia, mejor_ep = -1.0, 0, 0
    for ep in range(1, 16):
        model.train()
        for ea, eb, sa, sb, ca, cb, y in tqdm(dls["train"], desc=f"{nombre} {ep}",
                                              leave=False):
            ea, eb = ea.to(device), eb.to(device)
            sa, sb = sa.to(device), sb.to(device)
            ca, cb, y = ca.to(device), cb.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(ea, eb, sa, sb, ca, cb)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        pv, yv = m42.predecir(model, dls["val"], device)
        mv = m42.metricas(yv, pv)
        if mv["auc"] > mejor:
            mejor, paciencia, mejor_ep = mv["auc"], 0, ep
        else:
            paciencia += 1
            if paciencia >= 4:
                break
    pt, yt = m42.predecir(model, dls["test"], device)
    mt = m42.metricas(yt, pt)
    mt["variante"] = nombre
    mt["mejor_epoch"] = mejor_ep
    return mt


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pares = pd.read_csv(config.PROCESSED / "pares_scored.csv")[
        ["split", "label", "pos_a", "pos_b", "cat_a", "cat_b"]]
    embs = np.load(config.PROCESSED / "embeddings.npy")
    estilos = np.load(config.PROCESSED / "estilos_instancias.npy")
    filas = []
    for nombre, cfg in VARIANTES.items():
        print(f"\n=== {nombre} ===")
        m = entrenar(nombre, cfg, pares, embs, estilos, device)
        print(f"{nombre}: AUC={m['auc']:.3f} AP={m['ap']:.3f} F1={m['f1']:.3f}")
        filas.append(m)
    df = pd.DataFrame(filas)
    df.to_csv(config.METRICAS / "5_2_ablacion_compat.csv", index=False, encoding="utf-8")

    m3 = json.loads((config.METRICAS / "4_3_taste_metricas.json").read_text(encoding="utf-8"))
    r = m3["ranking"]
    reglas = pd.DataFrame([
        {"configuracion": "Taste sin reglas", "ndcg10": r["taste"]["ndcg10"],
         "hr10": r["taste"]["hr10"], "mrr10": r["taste"]["mrr10"],
         "violacion_top10": r["violacion_reglas_top10_taste"]},
        {"configuracion": "Taste + reglas", "ndcg10": r["taste_reglas"]["ndcg10"],
         "hr10": r["taste_reglas"]["hr10"], "mrr10": r["taste_reglas"]["mrr10"],
         "violacion_top10": r["violacion_reglas_top10_final"]},
    ])
    reglas.to_csv(config.METRICAS / "5_2_ablacion_reglas.csv", index=False, encoding="utf-8")

    resumen = {
        "compatibilidad": filas,
        "reglas": reglas.to_dict(orient="records"),
        "duracion_min": round((time.time() - t0) / 60, 2),
    }
    utils.json_dump(resumen, config.METRICAS / "5_2_ablacion.json")

    plt = utils.configurar_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.bar(df["variante"], df["auc"], color="#2f6f8f")
    for i, v in enumerate(df["auc"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC test")
    ax.set_title("Ablacion del modelo de compatibilidad")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    utils.guardar_figura(fig, "5_2_ablacion.png")

    lineas = [
        "# 5.2 Ablacion de componentes",
        "",
        "## Compatibilidad",
        "Se reentrenó el mismo MLP cambiando solo los bloques de entrada. Las "
        "variantes sin un bloque reciben ceros en esas dimensiones, por lo que la "
        "arquitectura y el coste de inferencia se mantienen comparables.",
        "",
        "| Variante | AUC | AP | Accuracy | F1 | Mejor época |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in filas:
        lineas.append(f"| {row['variante']} | {row['auc']:.3f} | {row['ap']:.3f} | "
                      f"{row['accuracy']:.3f} | {row['f1']:.3f} | {row['mejor_epoch']} |")
    lineas += [
        "",
        "La comparacion principal es **completo vs sin_cnn** y **completo vs "
        "sin_estilo**. Una caida grande indica que ese bloque aporta senal; una "
        "caida pequena indica redundancia entre bloques.",
        "",
        "![Ablacion](figuras/5_2_ablacion.png)",
        "",
        "## Reglas de gusto",
        "| Configuracion | NDCG@10 | HR@10 | MRR@10 | Violacion top-10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in reglas.itertuples(index=False):
        lineas.append(f"| {row.configuracion} | {row.ndcg10:.3f} | {row.hr10:.3f} | "
                      f"{row.mrr10:.3f} | {row.violacion_top10:.1%} |")
    lineas += [
        "",
        "Las reglas reducen la violacion sin destruir el ranking. El coste en NDCG "
        "es el precio de imponer coherencia; el punto de operacion elegido prioriza "
        "evitar combinaciones incoherentes en el top-10.",
        "",
        "## Limitaciones",
        "- La ablacion se hace sobre el subconjunto de 4.000 imagenes.",
        "- Las etiquetas de preferencia son sinteticas; la comparacion de reglas "
        "mide consistencia con el diseno, no satisfaccion de usuarios.",
        "- Los resultados dependen de una sola semilla; no se reporta intervalo de "
        "confianza por coste computacional.",
        "",
        f"Tiempo de ejecucion: {resumen['duracion_min']:.1f} min.",
    ]
    (config.REPORTS / "5.2_Reporte_comparativo_y_de_ablación.md").write_text("\n".join(lineas), encoding="utf-8")
    print(df[["variante", "auc", "ap", "accuracy", "f1"]].to_string(index=False))
    print(reglas.to_string(index=False))


if __name__ == "__main__":
    main()
