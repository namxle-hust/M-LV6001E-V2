from __future__ import annotations
import torch


def mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - tgt) ** 2)
