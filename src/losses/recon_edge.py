from __future__ import annotations
import torch, torch.nn as nn
from torch_geometric.utils import negative_sampling


class RelationDecoder(nn.Module):
    def __init__(
        self, in_src: int, in_dst: int, use_mlp: bool = False, hidden: int = 64
    ):
        super().__init__()
        if use_mlp:
            self.scorer = nn.Sequential(
                nn.Linear(in_src + in_dst, hidden), nn.ReLU(), nn.Linear(hidden, 1)
            )
            self.mode = "mlp"
        else:
            self.scorer = None
            self.mode = "dot"

    def forward(self, x_src, x_dst, edge_index):
        src, dst = edge_index
        pos = self._score(x_src[src], x_dst[dst]).squeeze(-1)
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=(x_src.size(0), x_dst.size(0)),
            num_neg_samples=src.numel(),
        )
        nsrc, ndst = neg_edge_index
        neg = self._score(x_src[nsrc], x_dst[ndst]).squeeze(-1)
        return pos, neg

    def _score(self, xs, xd):
        if self.mode == "dot":
            return (xs * xd).sum(dim=-1, keepdim=True)
        return self.scorer(torch.cat([xs, xd], dim=-1))


def edge_recon_losses(node_embeds: dict, data, use_mlp: bool = False):
    losses = {}
    total = 0.0
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    for rel in data.edge_types:
        edge_index = getattr(data[rel], "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            continue  # skip zero-edge relations safely

        x_src = node_embeds[rel[0]]
        x_dst = node_embeds[rel[2]]
        dec = RelationDecoder(x_src.size(-1), x_dst.size(-1), use_mlp=use_mlp).to(
            x_src.device
        )
        pos_logits, neg_logits = dec(x_src, x_dst, edge_index)
        if pos_logits.numel() == 0 and neg_logits.numel() == 0:
            continue
        loss = 0.0
        if pos_logits.numel() > 0:
            loss = loss + bce(pos_logits, torch.ones_like(pos_logits))
        if neg_logits.numel() > 0:
            loss = loss + bce(neg_logits, torch.zeros_like(neg_logits))
        losses[str(rel)] = loss
        total = total + loss

    losses["total"] = total
    return losses
