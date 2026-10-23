# -*- coding: utf-8 -*-
"""4.1 Entrenamiento de GarmentNet (CNN multi-cabeza)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             classification_report)
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src import utils
from src.dataset import GarmentDataset, cargar_meta_cnn, pesos_clase
from src.modelo_cnn import GarmentNet, contar_parametros

utils.ensure_dirs()
HEADS = ["silueta", "patron", "largo", "detalle"]


def evaluar(model, loader, device, criterion_cat, criterios_attr, n_clases=46):
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    attr_true = {h: [] for h in HEADS}
    attr_pred = {h: [] for h in HEADS}
    with torch.no_grad():
        for batch in loader:
            img = batch[0].to(device, non_blocking=True)
            ycat = batch[1].to(device, non_blocking=True)
            ys = {h: batch[2 + i].to(device, non_blocking=True) for i, h in enumerate(HEADS)}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(img)
                loss = criterion_cat(out["categoria"], ycat)
                for h in HEADS:
                    m = ys[h] != -100
                    if m.any():
                        loss = loss + 0.25 * criterios_attr[h](out[h][m], ys[h][m])
            losses.append(float(loss.detach().cpu()))
            prob = torch.softmax(out["categoria"].float(), dim=1).cpu().numpy()
            y_prob.append(prob)
            y_pred.extend(prob.argmax(axis=1).tolist())
            y_true.extend(ycat.cpu().numpy().tolist())
            for h in HEADS:
                m = ys[h] != -100
                if m.any():
                    attr_true[h].extend(ys[h][m].cpu().numpy().tolist())
                    attr_pred[h].extend(out[h][m].float().argmax(axis=1).cpu().numpy().tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prob = np.vstack(y_prob)
    top3 = np.argsort(-prob, axis=1)[:, :3]
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top3": float(np.mean([t in top3[i] for i, t in enumerate(y_true)])),
        "n": int(len(y_true)),
    }
    for h in HEADS:
        if attr_true[h]:
            metrics[f"acc_{h}"] = float(accuracy_score(attr_true[h], attr_pred[h]))
            metrics[f"f1_{h}"] = float(f1_score(attr_true[h], attr_pred[h],
                                                average="macro", zero_division=0))
        else:
            metrics[f"acc_{h}"] = 0.0
            metrics[f"f1_{h}"] = 0.0
    return metrics, y_true, y_pred, prob


def main():
    t0 = time.time()
    utils.set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    inst = pd.read_csv(config.PROCESSED / "instancias_split.csv")
    meta_train = cargar_meta_cnn(inst, "train")
    meta_val = cargar_meta_cnn(inst, "val")
    meta_test = cargar_meta_cnn(inst, "test")
    print(f"train={len(meta_train)} val={len(meta_val)} test={len(meta_test)}")

    memmap = config.PROCESSED / "recortes_144.npy"
    ds_train = GarmentDataset(meta_train, memmap, train=True)
    ds_val = GarmentDataset(meta_val, memmap, train=False)
    ds_test = GarmentDataset(meta_test, memmap, train=False)
    dl_train = DataLoader(ds_train, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    dl_val = DataLoader(ds_val, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    dl_test = DataLoader(ds_test, batch_size=config.BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

    cats = pd.read_csv(config.IDX / "categorias.csv")
    class_names = [str(x) for x in cats.sort_values("id")["name"].tolist()]
    model = GarmentNet(n_categorias=len(class_names)).to(device)
    print("Parametros:", f"{contar_parametros(model):,}")

    pesos = pesos_clase(meta_train["y_categoria"].to_numpy(), len(class_names)).to(device)
    crit_cat = nn.CrossEntropyLoss(weight=pesos)
    crit_attr = {h: nn.CrossEntropyLoss(ignore_index=-100) for h in HEADS}
    opt = torch.optim.AdamW(model.parameters(), lr=config.CNN_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.CNN_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    hist = []
    mejor = -1.0
    paciencia = 0
    for epoch in range(1, config.CNN_EPOCHS + 1):
        model.train()
        perdidas = []
        for batch in tqdm(dl_train, desc=f"epoch {epoch}/{config.CNN_EPOCHS}", leave=False):
            img = batch[0].to(device, non_blocking=True)
            ycat = batch[1].to(device, non_blocking=True)
            ys = {h: batch[2 + i].to(device, non_blocking=True) for i, h in enumerate(HEADS)}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(img)
                loss = crit_cat(out["categoria"], ycat)
                for h in HEADS:
                    m = ys[h] != -100
                    if m.any():
                        loss = loss + 0.25 * crit_attr[h](out[h][m], ys[h][m])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            perdidas.append(float(loss.detach().cpu()))
        sched.step()
        val, _, _, _ = evaluar(model, dl_val, device, crit_cat, crit_attr)
        row = {"epoch": epoch, "lr": float(opt.param_groups[0]["lr"]),
               "train_loss": float(np.mean(perdidas)), **{f"val_{k}": v for k, v in val.items()}}
        hist.append(row)
        print(f"epoch {epoch:02d} | train {row['train_loss']:.4f} | "
              f"val {val['loss']:.4f} | acc {val['accuracy']:.3f} | "
              f"macroF1 {val['macro_f1']:.3f} | top3 {val['top3']:.3f}")
        if val["macro_f1"] > mejor:
            mejor = val["macro_f1"]
            paciencia = 0
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "head_names": list(config.HEAD_ATTRS),
                "epoch": epoch,
                "val_metrics": val,
                "parametros": contar_parametros(model),
            }, config.MODELS / "garmentnet.pt")
        else:
            paciencia += 1
            if paciencia >= config.CNN_PATIENCE:
                print("Early stopping.")
                break

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(config.METRICAS / "cnn_historial.csv", index=False, encoding="utf-8")
    ckpt = torch.load(config.MODELS / "garmentnet.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test, y_true, y_pred, prob = evaluar(model, dl_test, device, crit_cat, crit_attr)
    test["mejor_epoch"] = int(ckpt["epoch"])
    test["parametros"] = int(contar_parametros(model))
    test["duracion_min"] = round((time.time() - t0) / 60, 2)
    utils.json_dump(test, config.METRICAS / "4_1_cnn_metricas.json")

    # Reporte por clase.
    rep = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, output_dict=True, zero_division=0)
    filas = []
    for i, nombre in enumerate(class_names):
        r = rep.get(nombre, {})
        filas.append({"categoria": nombre, "precision": r.get("precision", 0),
                      "recall": r.get("recall", 0), "f1": r.get("f1-score", 0),
                      "soporte": r.get("support", 0)})
    pd.DataFrame(filas).to_csv(config.METRICAS / "cnn_por_clase.csv", index=False, encoding="utf-8")

    # Matriz de confusion y curvas.
    plt = utils.configurar_matplotlib()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cmn = cm.astype(float) / np.maximum(1, cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=4.5)
    ax.set_yticklabels(class_names, fontsize=4.5)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion normalizada (test)")
    fig.colorbar(im, ax=ax, fraction=0.035)
    utils.guardar_figura(fig, "4_1_matriz_confusion.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="train")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], marker="o", label="val")
    axes[0].set_title("Perdida")
    axes[0].set_xlabel("Epoca")
    axes[0].legend(frameon=False)
    axes[1].plot(hist_df["epoch"], hist_df["val_accuracy"], marker="o", label="accuracy")
    axes[1].plot(hist_df["epoch"], hist_df["val_macro_f1"], marker="o", label="macro F1")
    axes[1].plot(hist_df["epoch"], hist_df["val_top3"], marker="o", label="top-3")
    axes[1].set_title("Metricas de validacion")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False)
    utils.guardar_figura(fig, "4_1_curvas.png")

    # Ejemplos de prediccion.
    mm = np.load(config.PROCESSED / "recortes_144.npy", mmap_mode="r")
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(meta_test), size=min(12, len(meta_test)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(9.0, 6.8))
    for ax, i in zip(axes.ravel(), idx):
        row = meta_test.iloc[int(i)]
        ax.imshow(np.asarray(mm[int(row["pos"])]))
        pred = class_names[int(prob[int(i)].argmax())]
        real = class_names[int(row["y_categoria"])]
        ax.set_title(f"real: {real[:18]}\npred: {pred[:18]}", fontsize=7)
        ax.axis("off")
    utils.guardar_figura(fig, "4_1_predicciones.png")

    lineas = [
        "# 4.1 Modelo CNN (GarmentNet)",
        "",
        "## Objetivo",
        "Entrenar GarmentNet para clasificar prendas por categoria y predecir "
        "cuatro grupos de atributos, generando un embedding de 128 dimensiones "
        "que alimenta los MLP de compatibilidad y personalizacion.",
        "",
        "## Arquitectura",
        "- CNN residual con tallo convolucional y cuatro etapas (24, 48, 96 y 208 canales).",
        f"- Parametros entrenables: **{test['parametros']:,}**.",
        "- Entrada: recorte RGB de 144x144, normalizado con media/desviacion 0.5.",
        "- Cabezas: categoria (46 clases), silueta (10), patron (9), largo (7), "
        "detalle (8) y embedding (128).",
        "- Perdida: entropia cruzada ponderada para categoria + 0.25 x cada cabeza "
        "de atributos (ignorando atributos ausentes).",
        "",
        "## Protocolo",
        f"- Train: **{len(meta_train):,}** instancias; val: **{len(meta_val):,}**; "
        f"test: **{len(meta_test):,}**.",
        f"- {config.CNN_EPOCHS} epocas maximas, AdamW (lr={config.CNN_LR}), cosine "
        "scheduling, AMP fp16, batch 64, early stopping (paciencia 5).",
        "- Aumento solo en train: volteo, rotacion/traslacion/escala y color jitter.",
        "",
        "## Resultados en test",
        f"- Accuracy: **{test['accuracy']:.3f}**.",
        f"- Macro F1: **{test['macro_f1']:.3f}**.",
        f"- Weighted F1: **{test['weighted_f1']:.3f}**.",
        f"- Top-3: **{test['top3']:.3f}**.",
        f"- Mejor epoca: **{test['mejor_epoch']}**.",
        "",
        "### Cabezas de atributos",
        "| Cabeza | Accuracy | Macro F1 |",
        "|---|---:|---:|",
    ]
    for h in HEADS:
        lineas.append(f"| {config.HEAD_ATTRS[h]['nombre']} | {test[f'acc_{h}']:.3f} | "
                      f"{test[f'f1_{h}']:.3f} |")
    lineas += [
        "",
        "## Evidencia visual",
        "![Curvas](figuras/4_1_curvas.png)",
        "",
        "![Matriz](figuras/4_1_matriz_confusion.png)",
        "",
        "![Predicciones](figuras/4_1_predicciones.png)",
        "",
        "## Archivos",
        "- `models/garmentnet.pt`: mejor checkpoint.",
        "- `reports/metricas/4_1_cnn_metricas.json`: metricas de test.",
        "- `reports/metricas/cnn_historial.csv`: historial por epoca.",
        "- `reports/metricas/cnn_por_clase.csv`: precision/recall/F1 por categoria.",
        "",
        "## Limitaciones",
        "- La muestra de 4.000 imagenes reduce el soporte de clases minoritarias; "
        "varias categorias tienen muy pocos ejemplos en test.",
        "- Las cabezas de atributos solo aprenden de instancias con atributos "
        "anotados; la cobertura es parcial (63,4 % en la muestra).",
        "- El embedding se entrena de forma auxiliar, sin una perdida metrica "
        "especifica; su calidad se evalua indirectamente en 4.2.",
        "",
        f"Tiempo de ejecucion: {test['duracion_min']:.1f} min.",
    ]
    (config.REPORTS / "4.1_Modelo_CNN_entrenado.md").write_text("\n".join(lineas), encoding="utf-8")
    print(json.dumps(test, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
