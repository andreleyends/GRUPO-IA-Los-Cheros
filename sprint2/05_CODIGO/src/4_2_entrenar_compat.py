# -*- coding: utf-8 -*-
"""4.2 MLP de compatibilidad sobre embeddings de GarmentNet."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import utils
from src.modelo_cnn import GarmentNet

utils.ensure_dirs()


class CompatMLP(nn.Module):
    def __init__(self, embed_dim=config.EMBED_DIM, cat_dim=16, style_dim=40,
                 n_cat=46, hidden=(256, 128, 64)):
        super().__init__()
        self.cat_emb = nn.Embedding(n_cat, cat_dim)
        inp = embed_dim * 3 + style_dim * 2 + cat_dim * 2
        capas = []
        prev = inp
        for h in hidden:
            capas += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                      nn.Dropout(0.2)]
            prev = h
        capas.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*capas)
        self.inp = inp

    def forward(self, ea, eb, sa, sb, ca, cb):
        x = torch.cat([ea, eb, torch.abs(ea - eb), sa, sb,
                       self.cat_emb(ca), self.cat_emb(cb)], dim=1)
        return self.mlp(x).squeeze(1)


def extraer_embeddings(inst: pd.DataFrame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(config.MODELS / "garmentnet.pt", map_location=device,
                      weights_only=False)
    model = GarmentNet(n_categorias=len(ckpt["class_names"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    meta = inst.sort_values("pos").reset_index(drop=True)
    mm = np.load(config.PROCESSED / "recortes_144.npy", mmap_mode="r")
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(meta), 256), desc="4.2 embeddings", unit="batch"):
            idx = meta.index[i:i + 256]
            imgs = []
            for j in idx:
                img = np.asarray(mm[int(meta.loc[j, "pos"])]).astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5
                imgs.append(torch.from_numpy(img.transpose(2, 0, 1)).float())
            batch = torch.stack(imgs).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(batch)
            embs.append(out["embedding"].float().cpu().numpy())
    arr = np.vstack(embs).astype(np.float32)
    np.save(config.PROCESSED / "embeddings.npy", arr)
    meta[["pos", "annotation_id", "image_id", "category_id", "category_name",
          "split"]].to_csv(config.PROCESSED / "embeddings_meta.csv",
                           index=False, encoding="utf-8")
    return meta, arr


def construir_tensores(pares: pd.DataFrame, embs: np.ndarray, estilos: np.ndarray):
    ea = embs[pares["pos_a"].to_numpy()]
    eb = embs[pares["pos_b"].to_numpy()]
    sa = estilos[pares["pos_a"].to_numpy()]
    sb = estilos[pares["pos_b"].to_numpy()]
    ca = torch.tensor(pares["cat_a"].to_numpy(), dtype=torch.long)
    cb = torch.tensor(pares["cat_b"].to_numpy(), dtype=torch.long)
    y = torch.tensor(pares["label"].to_numpy(), dtype=torch.float32)
    ds = TensorDataset(torch.tensor(ea), torch.tensor(eb), torch.tensor(sa),
                       torch.tensor(sb), ca, cb, y)
    return ds


def predecir(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for ea, eb, sa, sb, ca, cb, y in loader:
            ea, eb = ea.to(device), eb.to(device)
            sa, sb = sa.to(device), sb.to(device)
            ca, cb = ca.to(device), cb.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(ea, eb, sa, sb, ca, cb)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def metricas(y, p, prefijo=""):
    pred = (p >= 0.5).astype(int)
    out = {
        f"{prefijo}auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        f"{prefijo}ap": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        f"{prefijo}accuracy": float(accuracy_score(y, pred)),
        f"{prefijo}f1": float(f1_score(y, pred, zero_division=0)),
        f"{prefijo}precision": float(f1_score(y, pred, average="binary", zero_division=0)),
        f"{prefijo}recall": float((pred[y == 1].sum() / max(1, (y == 1).sum()))),
        f"{prefijo}n": int(len(y)),
    }
    # precision real (no F1) para evitar confusion semantica
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    out[f"{prefijo}precision"] = float(tp / max(1, tp + fp))
    return out


def main():
    t0 = time.time()
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inst = pd.read_csv(config.PROCESSED / "instancias_split.csv")
    if not (config.PROCESSED / "embeddings.npy").exists():
        extraer_embeddings(inst)
    embs = np.load(config.PROCESSED / "embeddings.npy")
    estilos = np.load(config.PROCESSED / "estilos_instancias.npy")
    pares_df = pd.read_csv(config.PROCESSED / "pares.csv")
    print(f"pares={len(pares_df)} embeddings={embs.shape} estilos={estilos.shape}")

    dss = {s: construir_tensores(pares_df[pares_df["split"] == s], embs, estilos)
           for s in ["train", "val", "test"]}
    dls = {s: DataLoader(dss[s], batch_size=256, shuffle=(s == "train")) for s in dss}
    model = CompatMLP().to(device)
    print("CompatMLP parametros:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=config.COMPAT_LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.COMPAT_EPOCHS)

    hist = []
    mejor = 0.0
    paciencia = 0
    for epoch in range(1, config.COMPAT_EPOCHS + 1):
        model.train()
        losses = []
        for ea, eb, sa, sb, ca, cb, y in tqdm(dls["train"], desc=f"compat {epoch}", leave=False):
            ea, eb = ea.to(device), eb.to(device)
            sa, sb = sa.to(device), sb.to(device)
            ca, cb, y = ca.to(device), cb.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(ea, eb, sa, sb, ca, cb)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        pv, yv = predecir(model, dls["val"], device)
        mv = metricas(yv, pv)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                     **{f"val_{k}": v for k, v in mv.items()}})
        print(f"epoch {epoch:02d} loss {hist[-1]['train_loss']:.4f} "
              f"val_auc {mv['auc']:.4f} val_f1 {mv['f1']:.4f}")
        if mv["auc"] > mejor:
            mejor = mv["auc"]
            paciencia = 0
            torch.save({"state_dict": model.state_dict(), "inp": model.inp,
                        "epoch": epoch, "val": mv,
                        "embed_dim": config.EMBED_DIM},
                       config.MODELS / "compat_head.pt")
        else:
            paciencia += 1
            if paciencia >= 6:
                break
    pd.DataFrame(hist).to_csv(config.METRICAS / "compat_historial.csv",
                              index=False, encoding="utf-8")
    ckpt = torch.load(config.MODELS / "compat_head.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    # Predicciones y metricas finales.
    pt, yt = predecir(model, dls["test"], device)
    mt = metricas(yt, pt)
    test_df = pares_df[pares_df["split"] == "test"].copy()
    test_df["score"] = pt
    por_origen = {}
    for origen, g in test_df.groupby("origen"):
        por_origen[origen] = metricas(g["label"].to_numpy(), g["score"].to_numpy())
        por_origen[origen]["score_medio"] = float(g["score"].mean())
    mt["por_origen"] = por_origen
    mt["mejor_epoch"] = int(ckpt["epoch"])
    mt["duracion_min"] = round((time.time() - t0) / 60, 2)
    utils.json_dump(mt, config.METRICAS / "4_2_compat_metricas.json")

    # Scores para el MLP de personalizacion.
    todas = pares_df.copy()
    ds_all = construir_tensores(todas, embs, estilos)
    dl_all = DataLoader(ds_all, batch_size=512, shuffle=False)
    p_all, _ = predecir(model, dl_all, device)
    todas["compat_score"] = p_all
    todas.to_csv(config.PROCESSED / "pares_scored.csv", index=False, encoding="utf-8")

    # Figuras.
    plt = utils.configurar_matplotlib()
    fpr, tpr, _ = roc_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(fpr, tpr, label=f"AUC={mt['auc']:.3f}", color="#2f6f8f")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdaderos positivos")
    ax.set_title("ROC - compatibilidad (test)")
    ax.legend(frameon=False)
    utils.guardar_figura(fig, "4_2_roc.png")

    prec, rec, _ = precision_recall_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(rec, prec, color="#6b8f71", label=f"AP={mt['ap']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - compatibilidad (test)")
    ax.legend(frameon=False)
    utils.guardar_figura(fig, "4_2_pr.png")

    cm = confusion_matrix(yt, (pt >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["no compatible", "compatible"])
    ax.set_yticks([0, 1], ["no compatible", "compatible"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion (test)")
    utils.guardar_figura(fig, "4_2_confusion.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for lab, color in [(1, "#2f6f8f"), (0, "#a24b3a")]:
        ax.hist(test_df.loc[test_df["label"] == lab, "score"], bins=30, alpha=0.6,
                label="compatible" if lab else "no compatible", color=color)
    ax.set_xlabel("Score de compatibilidad")
    ax.set_ylabel("Pares")
    ax.set_title("Distribucion de scores (test)")
    ax.legend(frameon=False)
    utils.guardar_figura(fig, "4_2_scores.png")

    lineas = [
        "# 4.2 MLP de compatibilidad",
        "",
        "## Objetivo",
        "Aprender un score de compatibilidad entre dos prendas usando el embedding "
        "de GarmentNet, la categoria y el vector de estilo de 40 dimensiones.",
        "",
        "## Diseno",
        "- Entrada: embedding CNN de A y B (128 cada uno), diferencia absoluta (128), "
        "estilos de A y B (40 cada uno) y embeddings de categoria (16 cada uno).",
        "- MLP 496 -> 256 -> 128 -> 64 -> 1, BatchNorm, ReLU, dropout 0.2.",
        "- Perdida BCE con `pos_weight=2` para compensar la proporcion 1:2.",
        "- Train/val/test heredados de 3.3; los pares sinteticos de apoyo solo "
        "estan en train y se evaluan aparte.",
        "",
        "## Resultados en test",
        f"- AUC: **{mt['auc']:.3f}**.",
        f"- AP: **{mt['ap']:.3f}**.",
        f"- Accuracy: **{mt['accuracy']:.3f}**.",
        f"- F1: **{mt['f1']:.3f}** (precision {mt['precision']:.3f}, recall {mt['recall']:.3f}).",
        f"- Mejor epoca: **{mt['mejor_epoch']}**.",
        "",
        "### Por origen",
        "El AUC no se calcula cuando un origen contiene una sola clase. Por eso la "
        "tabla reporta el score medio y la accuracy al umbral 0,5.",
        "",
        "| Origen | Pares | Score medio | Accuracy@0,5 |",
        "|---|---:|---:|---:|",
    ]
    for origen, m in por_origen.items():
        lineas.append(f"| {origen} | {m['n']:,} | {m['score_medio']:.3f} | "
                      f"{m['accuracy']:.3f} |")
    lineas += [
        "",
        "## Evidencia visual",
        "![ROC](figuras/4_2_roc.png)",
        "",
        "![PR](figuras/4_2_pr.png)",
        "",
        "![Confusion](figuras/4_2_confusion.png)",
        "",
        "![Scores](figuras/4_2_scores.png)",
        "",
        "## Archivos",
        "- `models/compat_head.pt`: checkpoint del MLP.",
        "- `data/processed/embeddings.npy`: embeddings CNN por instancia.",
        "- `data/processed/pares_scored.csv`: pares con score de compatibilidad.",
        "- `reports/metricas/4_2_compat_metricas.json`.",
        "",
        "## Limitaciones",
        "- El score se aprende de coocurrencia, no de juicios humanos de compatibilidad.",
        "- Los negativos aleatorios pueden ser combinaciones validas; el AUC mide "
        "separacion respecto al muestreo, no verdad absoluta.",
        "- El embedding CNN no tuvo una perdida metricamente supervisada; el MLP "
        "compensa con categoria y estilo.",
        "",
        f"Tiempo de ejecucion: {mt['duracion_min']:.1f} min.",
    ]
    (config.REPORTS / "4.2_Modelo_MLP_de_compatibilidad.md").write_text("\n".join(lineas), encoding="utf-8")
    print(json.dumps(mt, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
