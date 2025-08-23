from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import HeteroData


def build_patient_hetero_graph(
    patient_idx: int, tables: dict, edge_dfs: dict
) -> HeteroData:
    gene_ids = tables["gene_ids"]
    cpg_ids = tables["cpg_ids"]
    mirna_ids = tables["mirna_ids"]
    gene_id2idx = {g: i for i, g in enumerate(gene_ids)}
    cpg_id2idx = {c: i for i, c in enumerate(cpg_ids)}
    mir_id2idx = {m: i for i, m in enumerate(mirna_ids)}
    gexpr = tables["gene_expr"][:, patient_idx : patient_idx + 1]
    gcnv = tables["gene_cnv"][:, patient_idx : patient_idx + 1]
    gene_x = np.concatenate([gexpr, gcnv], axis=1).astype(np.float32)
    cpg = tables["cpg"][:, patient_idx : patient_idx + 1].astype(np.float32)
    mir = tables["mirna"][:, patient_idx : patient_idx + 1].astype(np.float32)
    data = HeteroData()
    data["gene"].x = torch.from_numpy(gene_x)
    data["cpg"].x = torch.from_numpy(cpg)
    data["mirna"].x = torch.from_numpy(mir)
    for (src, rel, dst), df in edge_dfs.items():
        if df is None:
            continue
        src_map = {"gene": gene_id2idx, "cpg": cpg_id2idx, "mirna": mir_id2idx}[src]
        dst_map = {"gene": gene_id2idx, "cpg": cpg_id2idx, "mirna": mir_id2idx}[dst]
        src_idx = []
        dst_idx = []
        w = []
        for row in df.itertuples(index=False):
            s = src_map.get(getattr(row, df.columns[0]))
            d = dst_map.get(getattr(row, df.columns[1]))
            if s is None or d is None:
                continue
            src_idx.append(s)
            dst_idx.append(d)
            w.append(getattr(row, df.columns[2]))
        if len(src_idx) == 0:
            src_idx = [0]
            dst_idx = [0]
            w = [0.0]

        data[(src, rel, dst)].edge_index = torch.tensor(
            [src_idx, dst_idx], dtype=torch.long
        )
        data[(src, rel, dst)].edge_weight = torch.tensor(w, dtype=torch.float32)
    return data
