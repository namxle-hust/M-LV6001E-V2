from __future__ import annotations
import os, pandas as pd


def _read_edge_csv(path: str, src_col: str, dst_col: str, weight_default: float = 1.0):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "weight" not in df.columns:
        df["weight"] = weight_default
    return df[[src_col, dst_col, "weight"]]


def load_edge_tables(cfg_data: dict) -> dict:
    edir = cfg_data["edges_dir"]
    rels = {}
    path_cpg = os.path.join(edir, cfg_data["gene_cpg_csv"])
    rels[("cpg", "maps_to", "gene")] = (
        _read_edge_csv(path_cpg, "cpg_id", "gene_id")
        if os.path.exists(path_cpg)
        else None
    )
    path_mir = os.path.join(edir, cfg_data["gene_mirna_csv"])
    rels[("mirna", "targets", "gene")] = (
        _read_edge_csv(path_mir, "mirna_id", "gene_id")
        if os.path.exists(path_mir)
        else None
    )
    if cfg_data.get("add_ppi", False):
        path_ppi = os.path.join(edir, cfg_data["gene_gene_csv"])
        rels[("gene", "ppi", "gene")] = (
            _read_edge_csv(path_ppi, "gene_id_src", "gene_id_dst")
            if os.path.exists(path_ppi)
            else None
        )
    return rels
