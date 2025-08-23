from __future__ import annotations
import torch, torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, GraphConv


def _make_conv(conv_name: str, in_ch: int, out_ch: int, heads: int = 1):
    if conv_name == "gatv2":
        return GATv2Conv(in_ch, out_ch // heads, heads=heads, add_self_loops=False)
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
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_adapters = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = layernorm
        for ntype in self.node_types:
            self.input_adapters[ntype] = nn.Identity()
        for _ in range(num_layers):
            convs = {
                (src, rel, dst): _make_conv(conv, hidden_dim, hidden_dim, heads=heads)
                for (src, rel, dst) in self.edge_types
            }
            self.layers.append(HeteroConv(convs, aggr="sum"))
            self.norms.append(
                nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types})
            )
        self.in_projs = nn.ModuleDict(
            {nt: nn.Linear(-1, hidden_dim, bias=False) for nt in self.node_types}
        )  # reset at runtime
        self.out_proj = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, out_dim) for nt in self.node_types}
        )

    def _reset_input(self, x_dict):
        for nt, x in x_dict.items():
            if isinstance(self.in_projs[nt], nn.Linear) and getattr(
                self.in_projs[nt], "in_features", -1
            ) == x.size(-1):
                continue
            self.in_projs[nt] = nn.Linear(
                x.size(-1), self.out_proj[nt].in_features, bias=False
            )

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        self._reset_input(x_dict)
        h = {nt: self.in_projs[nt](x) for nt, x in x_dict.items()}
        for l, conv in enumerate(self.layers):
            h = conv(h, edge_index_dict)
            h = {
                nt: (
                    torch.relu(self.norms[l][nt](v))
                    if self.layernorm
                    else torch.relu(v)
                )
                for nt, v in h.items()
            }
            h = {nt: self.dropout(v) for nt, v in h.items()}
        out = {nt: self.out_proj[nt](v) for nt, v in h.items()}
        return out
