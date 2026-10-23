# -*- coding: utf-8 -*-
"""GarmentNet: CNN residual multi-cabeza (~1.8 M parametros)."""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        ident = x if self.down is None else self.down(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + ident)


class GarmentNet(nn.Module):
    def __init__(self, widths=(24, 48, 96, 208), blocks=(2, 2, 2, 2),
                 n_categorias=46, embed_dim=config.EMBED_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        capas = []
        in_ch = widths[0]
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                stride = 2 if (i > 0 and j == 0) else 1
                capas.append(BasicBlock(in_ch, w, stride=stride))
                in_ch = w
        self.body = nn.Sequential(*capas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.head_categoria = nn.Linear(in_ch, n_categorias)
        self.head_silueta = nn.Linear(in_ch, 10)
        self.head_patron = nn.Linear(in_ch, 9)
        self.head_largo = nn.Linear(in_ch, 7)
        self.head_detalle = nn.Linear(in_ch, 8)
        self.head_embedding = nn.Linear(in_ch, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return {
            "categoria": self.head_categoria(x),
            "silueta": self.head_silueta(x),
            "patron": self.head_patron(x),
            "largo": self.head_largo(x),
            "detalle": self.head_detalle(x),
            "embedding": self.head_embedding(x),
        }


def contar_parametros(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
