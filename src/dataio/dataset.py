# src/dataio/dataset.py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path


class MultiOmicsDataset(Dataset):
    """PyTorch Geometric dataset for multi-omics patient graphs."""

    def __init__(self, config: dict, split: str = "train"):
        self.config = config
        self.split = split

        # Load features
        self.feature_loader = FeatureLoader(
            config["data"]["features_dir"], config["data"]["samples_file"]
        )

        # Load all feature files
        self.gene_features, self.gene_ids = self.feature_loader.load_features(
            config["data"]["genes_file"], "gene"
        )
        self.cpg_features, self.cpg_ids = self.feature_loader.load_features(
            config["data"]["cpgs_file"], "cpg"
        )
        self.mirna_features, self.mirna_ids = self.feature_loader.load_features(
            config["data"]["mirnas_file"], "mirna"
        )

        # Load edges
        self.edge_loader = EdgeLoader(config["data"]["edges_dir"])
        self.edges_data = {}

        self.edges_data["gene_cpg"] = self.edge_loader.load_edges(
            config["data"]["gene_cpg_file"], ("cpg", "maps_to", "gene")
        )
        self.edges_data["gene_mirna"] = self.edge_loader.load_edges(
            config["data"]["gene_mirna_file"], ("mirna", "targets", "gene")
        )

        if config["data"]["use_ppi_edges"]:
            self.edges_data["gene_gene"] = self.edge_loader.load_edges(
                config["data"]["gene_gene_file"], ("gene", "ppi", "gene")
            )

        # Build graph builder
        self.graph_builder = PatientGraphBuilder(
            self.gene_features,
            self.cpg_features,
            self.mirna_features,
            self.gene_ids,
            self.cpg_ids,
            self.mirna_ids,
            self.edges_data,
            config,
        )

        # Split patients into train/val
        self.patients = self.feature_loader.samples
        self._split_patients()

    def _split_patients(self):
        """Split patients into train/validation sets."""
        val_split = self.config["training"]["val_split"]
        n_val = int(len(self.patients) * val_split)

        # Use seed for reproducible split
        import random

        random.seed(self.config["seed"])
        shuffled_patients = self.patients.copy()
        random.shuffle(shuffled_patients)

        if self.split == "train":
            self.patient_ids = shuffled_patients[n_val:]
        else:  # validation
            self.patient_ids = shuffled_patients[:n_val]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> HeteroData:
        patient_id = self.patient_ids[idx]
        return self.graph_builder.build_graph(patient_id)

    def get_num_features(self) -> Dict[str, int]:
        """Get number of features for each node type."""
        return {
            "gene": self.gene_features.shape[0],
            "cpg": self.cpg_features.shape[0],
            "mirna": self.mirna_features.shape[0],
        }

    def get_node_counts(self) -> Dict[str, int]:
        """Get number of nodes for each type."""
        return {
            "gene": len(self.gene_ids),
            "cpg": len(self.cpg_ids),
            "mirna": len(self.mirna_ids),
        }


# src/dataio/collate.py
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader
from typing import List


def custom_collate(batch_list: List[HeteroData]) -> HeteroData:
    """
    Custom collate function for heterogeneous graphs.
    Each item in batch_list is a single patient's graph.
    """
    # Use PyG's Batch.from_data_list for hetero data
    batch = Batch.from_data_list(batch_list)

    # Store patient IDs
    batch.patient_ids = [data.patient_id for data in batch_list]

    return batch


def get_dataloader(
    dataset: MultiOmicsDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        num_workers=num_workers,
        pin_memory=True,
    )
