# -*- coding: utf-8 -*-
"""Dataset de recortes y aumentacion para GarmentNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config


class GarmentDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, memmap_path: Path, train: bool = False):
        self.meta = meta.reset_index(drop=True).copy()
        self.memmap_path = str(memmap_path)
        self.train = train
        self._mm = None
        if train:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=8, translate=(0.04, 0.04),
                                        scale=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                       saturation=0.12),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def _mmap(self):
        if self._mm is None:
            self._mm = np.load(self.memmap_path, mmap_mode="r")
        return self._mm

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        img = np.asarray(self._mmap()[int(row["pos"])])
        img = self.tf(img)
        return (
            img,
            int(row["y_categoria"]),
            int(row["y_silueta"]),
            int(row["y_patron"]),
            int(row["y_largo"]),
            int(row["y_detalle"]),
        )


def cargar_meta_cnn(instancias: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    from src.etiquetas import etiquetas_cabezas

    df = instancias.copy()
    cabezas = etiquetas_cabezas(df)
    df = pd.concat([df.reset_index(drop=True), cabezas.reset_index(drop=True)], axis=1)
    df["y_categoria"] = df["category_id"].astype(int)
    if split is not None:
        df = df[df["split"] == split].copy()
    return df.reset_index(drop=True)


def pesos_clase(y: np.ndarray, n_clases: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_clases).astype(float)
    pesos = np.ones(n_clases, dtype=np.float32)
    presentes = counts > 0
    pesos[presentes] = (1.0 / np.sqrt(counts[presentes])).astype(np.float32)
    if presentes.any():
        pesos = pesos / float(pesos[presentes].mean())
    return torch.tensor(pesos, dtype=torch.float32)
