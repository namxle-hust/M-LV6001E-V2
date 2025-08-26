"""
Edge loading utilities for heterogeneous graph construction.
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class EdgeLoader:
    """Load and process edge relationships between nodes."""

    def __init__(self, edges_dir: str, default_weight: float = 1.0):
        """
        Initialize edge loader.

        Args:
            edges_dir: Directory containing edge CSV files
            default_weight: Default edge weight if not specified
        """
        self.edges_dir = Path(edges_dir)
        self.default_weight = default_weight

    def load_cpg_gene_edges(
        self, edge_file: str, cpg_ids: List[str], gene_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load cpg to gene mapping edges.

        Args:
            edge_file: Path to gene_cpg.csv
            cpg_ids: List of cpg IDs
            gene_ids: List of gene IDs

        Returns:
            - Edge index tensor [2, n_edges]
            - Edge weight tensor [n_edges]
        """
        df = pd.read_csv(self.edges_dir / edge_file)

        # Create ID to index mappings
        cpg_to_idx = {cpg_id: idx for idx, cpg_id in enumerate(cpg_ids)}
        gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

        edges = []
        weights = []

        for _, row in df.iterrows():
            cpg_id = row["cpg_id"]
            gene_id = row["gene_id"]

            if cpg_id in cpg_to_idx and gene_id in gene_to_idx:
                edges.append([cpg_to_idx[cpg_id], gene_to_idx[gene_id]])

                # Use weight if provided, else default
                if "weight" in row and not pd.isna(row["weight"]):
                    weights.append(row["weight"])
                else:
                    weights.append(self.default_weight)

        if len(edges) == 0:
            print(f"Warning: No valid cpg-gene edges found")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            edge_index = torch.LongTensor(edges).t()
            edge_weight = torch.FloatTensor(weights)

        print(f"Loaded {len(edges)} cpg→gene edges")
        return edge_index, edge_weight

    def load_mirna_gene_edges(
        self, edge_file: str, mirna_ids: List[str], gene_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load mirna to gene target edges.

        Args:
            edge_file: Path to gene_mirna.csv
            mirna_ids: List of mirna IDs
            gene_ids: List of gene IDs

        Returns:
            - Edge index tensor [2, n_edges]
            - Edge weight tensor [n_edges]
        """
        df = pd.read_csv(self.edges_dir / edge_file)

        # Create ID to index mappings
        mirna_to_idx = {mirna_id: idx for idx, mirna_id in enumerate(mirna_ids)}
        gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

        edges = []
        weights = []

        for _, row in df.iterrows():
            mirna_id = row["mirna_id"]
            gene_id = row["gene_id"]

            if mirna_id in mirna_to_idx and gene_id in gene_to_idx:
                edges.append([mirna_to_idx[mirna_id], gene_to_idx[gene_id]])

                # Use weight if provided, else default
                if "weight" in row and not pd.isna(row["weight"]):
                    weights.append(row["weight"])
                else:
                    weights.append(self.default_weight)

        if len(edges) == 0:
            print(f"Warning: No valid mirna-gene edges found")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            edge_index = torch.LongTensor(edges).t()
            edge_weight = torch.FloatTensor(weights)

        print(f"Loaded {len(edges)} mirna→gene edges")
        return edge_index, edge_weight

    def load_gene_gene_edges(
        self, edge_file: str, gene_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load gene-gene PPI edges (optional).

        Args:
            edge_file: Path to gene_gene.csv
            gene_ids: List of gene IDs

        Returns:
            - Edge index tensor [2, n_edges]
            - Edge weight tensor [n_edges]
        """
        # Check if file exists
        edge_path = self.edges_dir / edge_file
        if not edge_path.exists():
            print(f"PPI file {edge_file} not found, skipping gene-gene edges")
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(
                0, dtype=torch.float
            )

        df = pd.read_csv(edge_path)

        # Create ID to index mapping
        gene_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}

        edges = []
        weights = []

        for _, row in df.iterrows():
            gene1 = row["gene1"] if "gene1" in row else row.iloc[0]
            gene2 = row["gene2"] if "gene2" in row else row.iloc[1]

            if gene1 in gene_to_idx and gene2 in gene_to_idx:
                idx1 = gene_to_idx[gene1]
                idx2 = gene_to_idx[gene2]

                # Add both directions for undirected edge
                edges.append([idx1, idx2])
                edges.append([idx2, idx1])

                # Use weight if provided, else default
                if "weight" in row and not pd.isna(row["weight"]):
                    weight = row["weight"]
                else:
                    weight = self.default_weight

                weights.append(weight)
                weights.append(weight)  # Same weight for both directions

        if len(edges) == 0:
            print(f"Warning: No valid gene-gene edges found")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            edge_index = torch.LongTensor(edges).t()
            edge_weight = torch.FloatTensor(weights)

            # Remove duplicate edges
            edge_index, edge_weight = remove_duplicate_edges(edge_index, edge_weight)

        print(f"Loaded {edge_index.shape[1]} gene↔gene edges")
        return edge_index, edge_weight

    def load_all_edges(self, config: dict, node_ids: Dict[str, List[str]]) -> Dict:
        """
        Load all edge types based on configuration.

        Args:
            config: Configuration dictionary
            node_ids: Dictionary of node IDs by type

        Returns:
            Dictionary containing all edge indices and weights
        """
        edges_dict = {}

        # Load cpg→gene edges
        cpg_gene_idx, cpg_gene_weight = self.load_cpg_gene_edges(
            config["data"]["gene_cpg_file"], node_ids["cpg"], node_ids["gene"]
        )
        edges_dict[("cpg", "maps_to", "gene")] = {
            "edge_index": cpg_gene_idx,
            "edge_weight": cpg_gene_weight,
        }

        # Load mirna→gene edges
        mirna_gene_idx, mirna_gene_weight = self.load_mirna_gene_edges(
            config["data"]["gene_mirna_file"], node_ids["mirna"], node_ids["gene"]
        )
        edges_dict[("mirna", "targets", "gene")] = {
            "edge_index": mirna_gene_idx,
            "edge_weight": mirna_gene_weight,
        }

        # Load gene↔gene edges if enabled
        if config["data"].get("use_ppi", False):
            gene_gene_idx, gene_gene_weight = self.load_gene_gene_edges(
                config["data"]["gene_gene_file"], node_ids["gene"]
            )
            edges_dict[("gene", "ppi", "gene")] = {
                "edge_index": gene_gene_idx,
                "edge_weight": gene_gene_weight,
            }

        # Add reverse edges for bidirectional message passing
        edges_dict[("gene", "mapped_by", "cpg")] = {
            "edge_index": torch.flip(cpg_gene_idx, dims=[0]),
            "edge_weight": cpg_gene_weight,
        }

        edges_dict[("gene", "targeted_by", "mirna")] = {
            "edge_index": torch.flip(mirna_gene_idx, dims=[0]),
            "edge_weight": mirna_gene_weight,
        }

        return edges_dict


def remove_duplicate_edges(
    edge_index: torch.Tensor, edge_weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove duplicate edges from edge list.

    Args:
        edge_index: Edge indices [2, n_edges]
        edge_weight: Edge weights [n_edges]

    Returns:
        Deduplicated edge index and weights
    """
    # Create unique edge identifiers
    edge_set = set()
    unique_indices = []

    for i in range(edge_index.shape[1]):
        edge = (edge_index[0, i].item(), edge_index[1, i].item())
        if edge not in edge_set:
            edge_set.add(edge)
            unique_indices.append(i)

    if len(unique_indices) < edge_index.shape[1]:
        print(f"Removed {edge_index.shape[1] - len(unique_indices)} duplicate edges")

    return edge_index[:, unique_indices], edge_weight[unique_indices]


def validate_edges(edges_dict: Dict, node_counts: Dict[str, int]) -> None:
    """
    Validate loaded edges for consistency.

    Args:
        edges_dict: Dictionary of edge data
        node_counts: Dictionary of node counts by type

    Raises:
        ValueError: If validation fails
    """
    for edge_type, edge_data in edges_dict.items():
        src_type, _, dst_type = edge_type
        edge_index = edge_data["edge_index"]

        if edge_index.shape[0] != 2:
            raise ValueError(
                f"Invalid edge index shape for {edge_type}: {edge_index.shape}"
            )

        if edge_index.shape[1] > 0:
            # Check that indices are within bounds
            max_src = edge_index[0].max().item()
            max_dst = edge_index[1].max().item()

            if max_src >= node_counts[src_type]:
                raise ValueError(
                    f"Source index out of bounds for {edge_type}: "
                    f"{max_src} >= {node_counts[src_type]}"
                )

            if max_dst >= node_counts[dst_type]:
                raise ValueError(
                    f"Destination index out of bounds for {edge_type}: "
                    f"{max_dst} >= {node_counts[dst_type]}"
                )

    print("Edge validation passed")
    for edge_type, edge_data in edges_dict.items():
        print(f"  {edge_type}: {edge_data['edge_index'].shape[1]} edges")
