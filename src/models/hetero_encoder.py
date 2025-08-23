from __future__ import annotations
import torch, torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, GraphConv


def _make_conv(conv_name: str, in_ch: int, out_ch: int, heads: int = 1):
    if conv_name == "gatv2":
        return GATv2Conv(
            in_ch,
            out_ch // heads if heads > 1 else out_ch,
            heads=heads,
            add_self_loops=False,
        )
    if conv_name == "sage":
        return SAGEConv(in_ch, out_ch)
    return GraphConv(in_ch, out_ch)


class HeteroGNN(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_dim=64,
        out_dim=64,
        num_layers=2,
        conv="gatv2",
        heads=2,
        dropout=0.1,
        layernorm=True,
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_dim, self.out_dim = hidden_dim, out_dim
        self.dropout = nn.Dropout(dropout)
        self.layernorm = layernorm

        # lazy per-type adapters: input_dim -> hidden_dim
        self.in_projs = nn.ModuleDict({nt: nn.Identity() for nt in self.node_types})

        # GNN layers on hidden_dim
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            convs = {
                (src, rel, dst): _make_conv(conv, hidden_dim, hidden_dim, heads=heads)
                for (src, rel, dst) in self.edge_types
            }
            self.layers.append(HeteroConv(convs, aggr="sum"))
            self.norms.append(
                nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types})
            )

        self.out_proj = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, out_dim) for nt in self.node_types}
        )

    def _reset_input_if_needed(self, x_dict):
        for nt, x in x_dict.items():
            mod = self.in_projs[nt]
            if (
                isinstance(mod, nn.Linear)
                and mod.in_features == x.size(-1)
                and mod.out_features == self.hidden_dim
            ):
                continue
            self.in_projs[nt] = nn.Linear(x.size(-1), self.hidden_dim, bias=False)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        self._reset_input_if_needed(x_dict)
        h = {nt: self.in_projs[nt](x) for nt, x in x_dict.items()}

        # Use only relations with at least one edge
        active = {
            rel: ei
            for rel, ei in edge_index_dict.items()
            if ei is not None and ei.numel() > 0
        }

        if active:
            for l, conv in enumerate(self.layers):
                h = conv(h, active)
                if self.layernorm:
                    h = {nt: torch.relu(self.norms[l][nt](v)) for nt, v in h.items()}
                else:
                    h = {nt: torch.relu(v) for nt, v in h.items()}
                h = {nt: self.dropout(v) for nt, v in h.items()}

        return {nt: self.out_proj[nt](v) for nt, v in h.items()}
