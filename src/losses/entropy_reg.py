from __future__ import annotations
import torch


def attention_entropy(alpha: torch.Tensor, eps: float = 1e-8):
    ent = -(alpha * (alpha + eps).log()).sum(dim=1).mean()
    return -ent
