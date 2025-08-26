"""
Feature loading utilities for multi-omics data.
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class FeatureLoader:
    """Load and process multi-omics feature data."""

    def __init__(self, features_dir: str):
        """
        Initialize feature loader.

        Args:
            features_dir: Directory containing feature TSV files
        """
        self.features_dir = Path(features_dir)

    def load_samples(self, samples_file: str) -> List[str]:
        """
        Load sample/patient IDs from file.

        Args:
            samples_file: Path to samples.txt file

        Returns:
            List of sample IDs
        """
        with open(samples_file, "r") as f:
            samples = [line.strip() for line in f if line.strip()]
        return samples

    def load_gene_features(
        self, expr_file: str, cnv_file: str, samples: List[str]
    ) -> Tuple[torch.Tensor, List[str], Dict]:
        """
        Load and concatenate mrna expression and cnv features for gene nodes.

        Args:
            expr_file: Path to gene expression TSV
            cnv_file: Path to cnv TSV
            samples: List of sample IDs to include

        Returns:
            - Concatenated features tensor [n_genes, n_samples, 2]
            - List of gene IDs
            - Dictionary with separate mrna and cnv tensors
        """
        # Load expression data
        expr_df = pd.read_csv(self.features_dir / expr_file, sep="\t", index_col=0)

        # Load cnv data
        cnv_df = pd.read_csv(self.features_dir / cnv_file, sep="\t", index_col=0)

        # Ensure same genes in same order
        common_genes = list(set(expr_df.index) & set(cnv_df.index))
        common_genes.sort()

        expr_df = expr_df.loc[common_genes]
        cnv_df = cnv_df.loc[common_genes]

        # Filter samples
        common_samples = list(set(samples) & set(expr_df.columns) & set(cnv_df.columns))
        if len(common_samples) < len(samples):
            print(
                f"Warning: Only {len(common_samples)}/{len(samples)} samples found in gene data"
            )

        expr_df = expr_df[common_samples]
        cnv_df = cnv_df[common_samples]

        # Convert to tensors
        expr_tensor = torch.FloatTensor(expr_df.values)  # [n_genes, n_samples]
        cnv_tensor = torch.FloatTensor(cnv_df.values)  # [n_genes, n_samples]

        # Concatenate as channels
        gene_features = torch.stack(
            [expr_tensor, cnv_tensor], dim=-1
        )  # [n_genes, n_samples, 2]

        # Also keep separate for returning
        separate_features = {"mrna": expr_tensor, "cnv": cnv_tensor}

        return gene_features, common_genes, separate_features

    def load_cpg_features(
        self, cpg_file: str, samples: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Load DNA methylation features for cpg nodes.

        Args:
            cpg_file: Path to cpg methylation TSV
            samples: List of sample IDs to include

        Returns:
            - Features tensor [n_cpgs, n_samples]
            - List of cpg IDs
        """
        cpg_df = pd.read_csv(self.features_dir / cpg_file, sep="\t", index_col=0)

        # Filter samples
        common_samples = list(set(samples) & set(cpg_df.columns))
        cpg_df = cpg_df[common_samples]

        cpg_tensor = torch.FloatTensor(cpg_df.values)

        return cpg_tensor, list(cpg_df.index)

    def load_mirna_features(
        self, mirna_file: str, samples: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Load mirna expression features.

        Args:
            mirna_file: Path to mirna expression TSV
            samples: List of sample IDs to include

        Returns:
            - Features tensor [n_mirnas, n_samples]
            - List of mirna IDs
        """
        mirna_df = pd.read_csv(self.features_dir / mirna_file, sep="\t", index_col=0)

        # Filter samples
        common_samples = list(set(samples) & set(mirna_df.columns))
        mirna_df = mirna_df[common_samples]

        mirna_tensor = torch.FloatTensor(mirna_df.values)

        return mirna_tensor, list(mirna_df.index)

    def load_all_features(self, config: dict) -> Dict:
        """
        Load all feature types based on configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary containing all loaded features and metadata
        """
        # Load samples
        samples = self.load_samples(config["data"]["samples_file"])

        # Load gene features (mrna + cnv)
        gene_features, gene_ids, gene_separate = self.load_gene_features(
            config["data"]["gene_expr_file"], config["data"]["gene_cnv_file"], samples
        )

        # Load cpg features
        cpg_features, cpg_ids = self.load_cpg_features(
            config["data"]["cpg_file"], samples
        )

        # Load mirna features
        mirna_features, mirna_ids = self.load_mirna_features(
            config["data"]["mirna_file"], samples
        )

        return {
            "samples": samples,
            "features": {
                "gene": gene_features,  # [n_genes, n_samples, 2]
                "cpg": cpg_features,  # [n_cpgs, n_samples]
                "mirna": mirna_features,  # [n_mirnas, n_samples]
            },
            "separate_gene_features": gene_separate,  # {'mrna': tensor, 'cnv': tensor}
            "node_ids": {"gene": gene_ids, "cpg": cpg_ids, "mirna": mirna_ids},
            "num_samples": len(samples),
            "num_nodes": {
                "gene": len(gene_ids),
                "cpg": len(cpg_ids),
                "mirna": len(mirna_ids),
            },
        }


def validate_features(features_dict: Dict) -> None:
    """
    Validate loaded features for consistency.

    Args:
        features_dict: Dictionary from load_all_features

    Raises:
        ValueError: If validation fails
    """
    # Check that all features have same number of samples
    n_samples = features_dict["num_samples"]

    gene_samples = features_dict["features"]["gene"].shape[1]
    cpg_samples = features_dict["features"]["cpg"].shape[1]
    mirna_samples = features_dict["features"]["mirna"].shape[1]

    if not (gene_samples == cpg_samples == mirna_samples == n_samples):
        raise ValueError(
            f"Sample count mismatch: gene={gene_samples}, cpg={cpg_samples}, "
            f"mirna={mirna_samples}, expected={n_samples}"
        )

    # Check for NaN values
    for node_type, features in features_dict["features"].items():
        if torch.isnan(features).any():
            raise ValueError(f"NaN values found in {node_type} features")

    print(f"Feature validation passed:")
    print(f"  - Samples: {n_samples}")
    print(f"  - gene nodes: {features_dict['num_nodes']['gene']}")
    print(f"  - cpg nodes: {features_dict['num_nodes']['cpg']}")
    print(f"  - mirna nodes: {features_dict['num_nodes']['mirna']}")
