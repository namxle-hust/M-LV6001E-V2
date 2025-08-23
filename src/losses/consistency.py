from __future__ import annotations
import torch


def consistency_loss(h, modality_vecs: dict):
    loss = 0.0
    for z in modality_vecs.values():
        loss = loss + torch.mean((h - z) ** 2)
    return loss / max(len(modality_vecs), 1)
