from __future__ import annotations
import torch, torch.nn as nn


class ModalityAttention(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, modality_vectors: dict[str, torch.Tensor]):
        names = list(modality_vectors.keys())
        Z = torch.stack([modality_vectors[n] for n in names], dim=1)  # [B,M,D]
        B, M, D = Z.shape
        scores = self.scorer(Z.view(B * M, D)).view(B, M)
        alpha = torch.softmax(scores, dim=1)
        h = (alpha.unsqueeze(-1) * Z).sum(dim=1)
        return h, alpha, names
