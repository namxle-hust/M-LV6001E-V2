from __future__ import annotations
import torch, torch.nn as nn
from torch_geometric.nn import global_mean_pool


class MeanPool(nn.Module):
    def forward(self, x, batch):
        return global_mean_pool(x, batch)


class AttnPool(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, batch):
        scores = self.scorer(x).squeeze(-1)
        max_batch = int(batch.max().item()) if batch.numel() > 0 else -1
        outs = []
        for i in range(max_batch + 1):
            mask = batch == i
            if mask.sum() == 0:
                outs.append(torch.zeros(x.size(1), device=x.device))
                continue
            s = torch.softmax(scores[mask], dim=0).unsqueeze(-1)
            outs.append((x[mask] * s).sum(dim=0))
        return torch.stack(outs, dim=0)


class GeneHeads(nn.Module):
    """Two separate heads to split gene-pooled vector into mRNA and CNV modality vectors."""

    def __init__(self, in_dim, out_dim):
        super().__init__()

        def head():
            return nn.Sequential(
                nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim)
            )

        self.mrna = head()
        self.cnv = head()

    def forward(self, gene_vec):
        return self.mrna(gene_vec), self.cnv(gene_vec)


class ModalityPooling(nn.Module):
    def __init__(self, node_out_dims: dict, cfg: dict):
        super().__init__()
        dim = cfg["modalities"]["dim"]
        ptype = cfg["pooling"]["type"]
        if ptype == "attn":
            self.pools = {
                "gene": AttnPool(
                    node_out_dims["gene"], cfg["pooling"].get("attn_hidden", 64)
                ),
                "cpg": AttnPool(
                    node_out_dims["cpg"], cfg["pooling"].get("attn_hidden", 64)
                ),
                "mirna": AttnPool(
                    node_out_dims["mirna"], cfg["pooling"].get("attn_hidden", 64)
                ),
            }
        else:
            self.pools = {"gene": MeanPool(), "cpg": MeanPool(), "mirna": MeanPool()}
        self.gene_heads = GeneHeads(node_out_dims["gene"], dim)
        self.lin_cpg = nn.Linear(node_out_dims["cpg"], dim)
        self.lin_mir = nn.Linear(node_out_dims["mirna"], dim)

    def forward(self, node_embeds: dict, batch_dict: dict):
        g_pool = self.pools["gene"](node_embeds["gene"], batch_dict["gene"])
        c_pool = self.pools["cpg"](node_embeds["cpg"], batch_dict["cpg"])
        m_pool = self.pools["mirna"](node_embeds["mirna"], batch_dict["mirna"])
        z_mrna, z_cnv = self.gene_heads(g_pool)
        z_dnam = self.lin_cpg(c_pool)
        z_mir = self.lin_mir(m_pool)
        return {"mRNA": z_mrna, "CNV": z_cnv, "DNAmeth": z_dnam, "miRNA": z_mir}
