"""
Build heterogeneous patient graphs from multi-omics data.
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple
import numpy as np


class PatientGraphBuilder:
    """Build heterogeneous graphs for individual patients."""

    def __init__(self, features_dict: Dict, edges_dict: Dict):
        """
        Initialize graph builder.

        Args:
            features_dict: Dictionary containing all features and metadata
            edges_dict: Dictionary containing all edge information
        """
        self.features_dict = features_dict
        self.edges_dict = edges_dict
        self.num_samples = features_dict["num_samples"]

    def build_patient_graph(self, patient_idx: int) -> HeteroData:
        """
        Build a heterogeneous graph for a single patient.

        Args:
            patient_idx: Index of the patient (0 to num_samples-1)

        Returns:
            HeteroData object for the patient
        """
        data = HeteroData()

        # Add node features for this patient
        # Gene nodes: Extract patient column from [n_genes, n_samples, 2]
        gene_features = self.features_dict["features"]["gene"][:, patient_idx, :]
        data["gene"].x = gene_features  # [n_genes, 2] (mrna + cnv channels)

        # cpg nodes: Extract patient column from [n_cpgs, n_samples]
        cpg_features = self.features_dict["features"]["cpg"][:, patient_idx]
        data["cpg"].x = cpg_features.unsqueeze(-1)  # [n_cpgs, 1]

        # mirna nodes: Extract patient column from [n_mirnas, n_samples]
        mirna_features = self.features_dict["features"]["mirna"][:, patient_idx]
        data["mirna"].x = mirna_features.unsqueeze(-1)  # [n_mirnas, 1]

        # Store node IDs as metadata
        data["gene"].node_ids = self.features_dict["node_ids"]["gene"]
        data["cpg"].node_ids = self.features_dict["node_ids"]["cpg"]
        data["mirna"].node_ids = self.features_dict["node_ids"]["mirna"]

        # Add edges (same for all patients)
        for edge_type, edge_data in self.edges_dict.items():
            if edge_data["edge_index"].shape[1] > 0:  # Only add if edges exist
                data[edge_type].edge_index = edge_data["edge_index"]
                data[edge_type].edge_weight = edge_data["edge_weight"]

        # Add patient metadata
        data.patient_id = self.features_dict["samples"][patient_idx]
        data.patient_idx = patient_idx

        # Store separate gene features for later use
        data.gene_mrna = self.features_dict["separate_gene_features"]["mrna"][
            :, patient_idx
        ]
        data.gene_cnv = self.features_dict["separate_gene_features"]["cnv"][
            :, patient_idx
        ]

        return data

    def build_all_graphs(self) -> List[HeteroData]:
        """
        Build graphs for all patients.

        Returns:
            List of HeteroData objects
        """
        graphs = []
        for i in range(self.num_samples):
            graph = self.build_patient_graph(i)
            graphs.append(graph)
        return graphs

    def get_graph_statistics(self, graph: HeteroData) -> Dict:
        """
        Get statistics for a patient graph.

        Args:
            graph: HeteroData object

        Returns:
            Dictionary of statistics
        """
        stats = {
            "patient_id": graph.patient_id,
            "num_nodes": {},
            "num_edges": {},
            "node_features": {},
        }

        # Node counts
        for node_type in graph.node_types:
            stats["num_nodes"][node_type] = graph[node_type].x.shape[0]
            stats["node_features"][node_type] = {
                "shape": list(graph[node_type].x.shape),
                "min": graph[node_type].x.min().item(),
                "max": graph[node_type].x.max().item(),
                "mean": graph[node_type].x.mean().item(),
                "std": graph[node_type].x.std().item(),
            }

        # Edge counts
        for edge_type in graph.edge_types:
            if hasattr(graph[edge_type], "edge_index"):
                stats["num_edges"][edge_type] = graph[edge_type].edge_index.shape[1]

        return stats


def create_batch_from_graphs(graphs: List[HeteroData]) -> HeteroData:
    """
    Create a batched graph from list of patient graphs.

    Args:
        graphs: List of HeteroData objects

    Returns:
        Batched HeteroData object
    """
    from torch_geometric.data import Batch

    # Create batch
    batch = Batch.from_data_list(graphs)

    # Add batch-level metadata
    batch.patient_ids = [g.patient_id for g in graphs]
    batch.patient_indices = [g.patient_idx for g in graphs]

    return batch


def split_patients(
    num_patients: int, val_split: float = 0.2, test_split: float = 0.1, seed: int = 42
) -> Dict[str, List[int]]:
    """
    Split patient indices into train/val/test sets.

    Args:
        num_patients: Total number of patients
        val_split: Fraction for validation
        test_split: Fraction for test
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' indices
    """
    np.random.seed(seed)
    indices = np.arange(num_patients)
    np.random.shuffle(indices)

    n_test = int(num_patients * test_split)
    n_val = int(num_patients * val_split)
    n_train = num_patients - n_test - n_val

    splits = {
        "train": indices[:n_train].tolist(),
        "val": indices[n_train : n_train + n_val].tolist(),
        "test": indices[n_train + n_val :].tolist(),
    }

    print(f"Data split: train={n_train}, val={n_val}, test={n_test}")

    return splits


class GraphNormalizer:
    """Normalize graph features across patients."""

    def __init__(self):
        self.stats = {}

    def fit(self, graphs: List[HeteroData]):
        """
        Compute normalization statistics from training graphs.

        Args:
            graphs: List of training graphs
        """
        # Collect all features by node type
        features_by_type = {"gene": [], "cpg": [], "mirna": []}

        for graph in graphs:
            for node_type in features_by_type.keys():
                features_by_type[node_type].append(graph[node_type].x)

        # Compute statistics
        for node_type, features_list in features_by_type.items():
            if features_list:
                all_features = torch.cat(features_list, dim=0)
                self.stats[node_type] = {
                    "mean": all_features.mean(dim=0),
                    "std": all_features.std(dim=0) + 1e-8,  # Avoid division by zero
                }

    def transform(self, graph: HeteroData) -> HeteroData:
        """
        Apply normalization to a graph.

        Args:
            graph: Graph to normalize

        Returns:
            Normalized graph
        """
        for node_type in self.stats.keys():
            if node_type in graph.node_types:
                mean = self.stats[node_type]["mean"]
                std = self.stats[node_type]["std"]
                graph[node_type].x = (graph[node_type].x - mean) / std

        return graph

    def fit_transform(self, graphs: List[HeteroData]) -> List[HeteroData]:
        """
        Fit and transform graphs.

        Args:
            graphs: List of graphs

        Returns:
            List of normalized graphs
        """
        self.fit(graphs)
        return [self.transform(g) for g in graphs]
