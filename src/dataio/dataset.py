"""
PyTorch Geometric dataset for patient graphs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData, Batch
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np


class PatientGraphDataset(Dataset):
    """Dataset for patient heterogeneous graphs."""

    def __init__(self, graphs: List[HeteroData], transform: Optional[Callable] = None):
        """
        Initialize patient graph dataset.

        Args:
            graphs: List of patient HeteroData graphs
            transform: Optional transform to apply to graphs
        """
        self.graphs = graphs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> HeteroData:
        graph = self.graphs[idx]

        if self.transform is not None:
            graph = self.transform(graph)

        return graph

    def get_patient_id(self, idx: int) -> str:
        """Get patient ID for a given index."""
        return self.graphs[idx].patient_id

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "num_patients": len(self.graphs),
            "num_nodes": {},
            "num_edges": {},
            "feature_dims": {},
        }

        # Aggregate statistics
        for node_type in self.graphs[0].node_types:
            node_counts = [g[node_type].x.shape[0] for g in self.graphs]
            stats["num_nodes"][node_type] = {
                "mean": np.mean(node_counts),
                "std": np.std(node_counts),
                "min": np.min(node_counts),
                "max": np.max(node_counts),
            }
            stats["feature_dims"][node_type] = self.graphs[0][node_type].x.shape[1]

        for edge_type in self.graphs[0].edge_types:
            if hasattr(self.graphs[0][edge_type], "edge_index"):
                edge_counts = [g[edge_type].edge_index.shape[1] for g in self.graphs]
                stats["num_edges"][edge_type] = {
                    "mean": np.mean(edge_counts),
                    "std": np.std(edge_counts),
                    "min": np.min(edge_counts),
                    "max": np.max(edge_counts),
                }

        return stats


def custom_collate_fn(batch: List[HeteroData]) -> HeteroData:
    """
    Custom collate function for batching heterogeneous graphs.

    Args:
        batch: List of HeteroData graphs

    Returns:
        Batched HeteroData
    """
    # Use PyG's batch functionality
    batched_data = Batch.from_data_list(batch)

    # Add custom batch-level attributes
    batched_data.patient_ids = [g.patient_id for g in batch]
    batched_data.patient_indices = torch.tensor([g.patient_idx for g in batch])

    # Combine gene_mrna and gene_cnv features
    if hasattr(batch[0], "gene_mrna"):
        gene_mrna_list = [g.gene_mrna for g in batch]
        gene_cnv_list = [g.gene_cnv for g in batch]

        # Get batch assignment for gene nodes
        batch_assign = batched_data["gene"].batch

        # Stack features
        batched_data.gene_mrna_batched = torch.cat(gene_mrna_list, dim=0)
        batched_data.gene_cnv_batched = torch.cat(gene_cnv_list, dim=0)

    return batched_data


class MultiModalDataModule:
    """Data module for handling train/val/test splits with K-fold support."""

    def __init__(self, all_graphs: List[HeteroData], config: dict, seed: int = 42):
        """
        Initialize data module.

        Args:
            all_graphs: All patient graphs
            config: Configuration dictionary
            seed: Random seed for splitting
        """
        self.all_graphs = all_graphs
        self.config = config
        self.seed = seed

        # K-fold specific attributes
        self.kfold_splits = None
        self.current_fold = None

        # Split data for standard training
        self.train_graphs, self.val_graphs, self.test_graphs = self._split_data()

        # Create datasets
        self.train_dataset = PatientGraphDataset(self.train_graphs)
        self.val_dataset = PatientGraphDataset(self.val_graphs)
        self.test_dataset = PatientGraphDataset(self.test_graphs)

    def _split_data(self) -> Tuple:
        """Split data into train/val/test sets."""
        n_total = len(self.all_graphs)
        val_split = self.config["training"].get("validation_split", 0.2)
        test_split = self.config["training"].get("test_split", 0.1)

        # Set seed for reproducibility
        np.random.seed(self.seed)
        indices = np.arange(n_total)
        np.random.shuffle(indices)

        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_test - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_graphs = [self.all_graphs[i] for i in train_idx]
        val_graphs = [self.all_graphs[i] for i in val_idx]
        test_graphs = [self.all_graphs[i] for i in test_idx]

        print(
            f"Data split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}"
        )

        return train_graphs, val_graphs, test_graphs

    def setup_kfold(self, n_folds: int = 5):
        """Setup K-fold cross-validation splits."""
        from ..utils.kfold import create_kfold_splits

        n_total = len(self.all_graphs)
        self.kfold_splits = create_kfold_splits(n_total, n_folds, self.seed)
        print(f"Created {n_folds}-fold splits with {n_total} samples")

    def set_fold(self, fold_idx: int):
        """Set current fold for training."""
        if self.kfold_splits is None:
            raise ValueError("Must call setup_kfold first")

        train_idx, val_idx = self.kfold_splits[fold_idx]
        self.train_graphs = [self.all_graphs[i] for i in train_idx]
        self.val_graphs = [self.all_graphs[i] for i in val_idx]

        # Update datasets
        self.train_dataset = PatientGraphDataset(self.train_graphs)
        self.val_dataset = PatientGraphDataset(self.val_graphs)

        self.current_fold = fold_idx
        print(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")

    def get_dataloader(
        self,
        dataset: PatientGraphDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create a dataloader for a dataset.

        Args:
            dataset: PatientGraphDataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory

        Returns:
            DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return self.get_dataloader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=self.config["training"]["pin_memory"],
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return self.get_dataloader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=self.config["training"]["pin_memory"],
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return self.get_dataloader(
            self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=self.config["training"]["pin_memory"],
        )
