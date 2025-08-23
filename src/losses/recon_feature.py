from __future__ import annotations
import torch, torch.nn as nn


class FeatureDecoders(torch.nn.Module):
    def __init__(self, node_embed_dims: dict, orig_feat_dims: dict, hidden: int = 64):
        super().__init__()
        self.gene_mrna = nn.Linear(node_embed_dims["gene"], 1)
        self.gene_cnv = nn.Linear(node_embed_dims["gene"], 1)
        self.cpg = nn.Linear(node_embed_dims["cpg"], orig_feat_dims["cpg"])
        self.mirna = nn.Linear(node_embed_dims["mirna"], orig_feat_dims["mirna"])

    def forward(self, embeds: dict, batch):
        return {
            "gene_mrna": self.gene_mrna(embeds["gene"]),
            "gene_cnv": self.gene_cnv(embeds["gene"]),
            "cpg": self.cpg(embeds["cpg"]),
            "mirna": self.mirna(embeds["mirna"]),
        }


def recon_feature_loss(outputs: dict, inputs: dict, batch_dict: dict):
    mse = torch.nn.functional.mse_loss
    gene_x = inputs["gene"]
    mrna_tgt = gene_x[:, :1]
    cnv_tgt = gene_x[:, 1:2]
    losses = {
        "gene": mse(outputs["gene_mrna"], mrna_tgt) + mse(outputs["gene_cnv"], cnv_tgt),
        "mRNA": mse(outputs["gene_mrna"], mrna_tgt),
        "CNV": mse(outputs["gene_cnv"], cnv_tgt),
        "DNAmeth": mse(outputs["cpg"], inputs["cpg"]),
        "miRNA": mse(outputs["mirna"], inputs["mirna"]),
    }
    losses["total"] = losses["gene"] + losses["DNAmeth"] + losses["miRNA"]
    return losses
