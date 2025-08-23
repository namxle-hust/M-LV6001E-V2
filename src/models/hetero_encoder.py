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
