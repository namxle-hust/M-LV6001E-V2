from __future__ import annotations
import os, pandas as pd, numpy as np
from typing import Tuple


def _read_tsv_matrix(path: str) -> Tuple[list[str], list[str], np.ndarray]:
    df = pd.read_csv(path, sep="\t")
    df = df.set_index(df.columns[0])
    rows = df.index.astype(str).tolist()
    cols = df.columns.astype(str).tolist()
    mat = df.values.astype(np.float32)
    return rows, cols, mat


def load_feature_tables(cfg_data: dict):
    fdir = cfg_data["features_dir"]
    gexpr_rows, gexpr_cols, gexpr = _read_tsv_matrix(
        os.path.join(fdir, cfg_data["genes_expr_tsv"])
    )
    gcnv_rows, gcnv_cols, gcnv = _read_tsv_matrix(
        os.path.join(fdir, cfg_data["genes_cnv_tsv"])
    )
    assert gexpr_rows == gcnv_rows
    assert gexpr_cols == gcnv_cols
    cpg_rows, cpg_cols, cpg = _read_tsv_matrix(os.path.join(fdir, cfg_data["cpgs_tsv"]))
    mir_rows, mir_cols, mir = _read_tsv_matrix(
        os.path.join(fdir, cfg_data["mirnas_tsv"])
    )
    assert gexpr_cols == cpg_cols == mir_cols
    return {
        "patients": gexpr_cols,
        "gene_ids": gexpr_rows,
        "cpg_ids": cpg_rows,
        "mirna_ids": mir_rows,
        "gene_expr": gexpr,
        "gene_cnv": gcnv,
        "cpg": cpg,
        "mirna": mir,
    }
