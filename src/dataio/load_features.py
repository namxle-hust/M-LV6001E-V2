# src/dataio/load_features.py
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class FeatureLoader:
    """Load and process multi-omic feature data."""

    def __init__(self, features_dir: str, samples_file: str):
        self.features_dir = Path(features_dir)
        self.samples_file = Path(samples_file)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[str]:
        """Load patient IDs from samples.txt."""
        with open(self.samples_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def load_features(
        self, filename: str, node_type: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load feature TSV file.
        Returns: (dataframe, feature_ids)
        """
        filepath = self.features_dir / filename
        df = pd.read_csv(filepath, sep="\t", index_col=0)

        # Ensure all samples are present
        missing_samples = set(self.samples) - set(df.columns)
        if missing_samples:
            print(f"Warning: Missing samples in {node_type}: {missing_samples}")
            # Add missing samples with zeros
            for sample in missing_samples:
                df[sample] = 0.0

        # Reorder columns to match samples order
        df = df[self.samples]

        # Normalize and impute
        df = self._normalize_and_impute(df)

        feature_ids = df.index.tolist()
        return df, feature_ids

    def _normalize_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features and impute missing values."""
        # Standard normalization per feature
        df = (df - df.mean(axis=1).values.reshape(-1, 1)) / (
            df.std(axis=1).values.reshape(-1, 1) + 1e-8
        )

        # Impute missing values with 0
        df = df.fillna(0)

        return df

    def get_patient_features(self, df: pd.DataFrame, patient_id: str) -> torch.Tensor:
        """Extract features for a specific patient."""
        if patient_id not in df.columns:
            raise ValueError(f"Patient {patient_id} not found in features")

        features = df[patient_id].values
        return torch.tensor(features, dtype=torch.float32)


# src/dataio/load_edges.py
class EdgeLoader:
    """Load edge relationships from CSV files."""

    def __init__(self, edges_dir: str):
        self.edges_dir = Path(edges_dir)

    def load_edges(
        self, filename: str, edge_type: Tuple[str, str, str]
    ) -> pd.DataFrame:
        """
        Load edge CSV file.
        Expected columns: source_id, target_id, weight (optional)
        """
        filepath = self.edges_dir / filename

        if not filepath.exists():
            print(f"Warning: Edge file {filename} not found")
            return pd.DataFrame(columns=["source", "target", "weight"])

        df = pd.read_csv(filepath)

        # Rename columns for consistency
        column_mapping = {df.columns[0]: "source", df.columns[1]: "target"}
        df = df.rename(columns=column_mapping)

        # Add weight column if not present
        if "weight" not in df.columns and len(df.columns) > 2:
            df["weight"] = df.iloc[:, 2]
        elif "weight" not in df.columns:
            df["weight"] = 1.0

        return df[["source", "target", "weight"]]


# src/dataio/build_patient_graph.py
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional


class PatientGraphBuilder:
    """Build heterogeneous graph for each patient."""

    def __init__(
        self,
        gene_features: pd.DataFrame,
        cpg_features: pd.DataFrame,
        mirna_features: pd.DataFrame,
        gene_ids: List[str],
        cpg_ids: List[str],
        mirna_ids: List[str],
        edges_data: Dict[str, pd.DataFrame],
        config: dict,
    ):

        self.gene_features = gene_features
        self.cpg_features = cpg_features
        self.mirna_features = mirna_features

        # Create ID to index mappings
        self.gene_to_idx = {gene: i for i, gene in enumerate(gene_ids)}
        self.cpg_to_idx = {cpg: i for i, cpg in enumerate(cpg_ids)}
        self.mirna_to_idx = {mirna: i for i, mirna in enumerate(mirna_ids)}

        self.edges_data = edges_data
        self.config = config

    def build_graph(self, patient_id: str) -> HeteroData:
        """Build heterogeneous graph for a single patient."""
        data = HeteroData()

        # Add node features
        data["gene"].x = self._get_patient_features(self.gene_features, patient_id)
        data["cpg"].x = self._get_patient_features(self.cpg_features, patient_id)
        data["mirna"].x = self._get_patient_features(self.mirna_features, patient_id)

        # Store number of nodes
        data["gene"].num_nodes = len(self.gene_to_idx)
        data["cpg"].num_nodes = len(self.cpg_to_idx)
        data["mirna"].num_nodes = len(self.mirna_to_idx)

        # Add edges
        # CpG -> Gene edges
        if "gene_cpg" in self.edges_data:
            edge_index, edge_weight = self._build_edges(
                self.edges_data["gene_cpg"], self.cpg_to_idx, self.gene_to_idx
            )
            data["cpg", "maps_to", "gene"].edge_index = edge_index
            data["cpg", "maps_to", "gene"].edge_weight = edge_weight

        # miRNA -> Gene edges
        if "gene_mirna" in self.edges_data:
            edge_index, edge_weight = self._build_edges(
                self.edges_data["gene_mirna"], self.mirna_to_idx, self.gene_to_idx
            )
            data["mirna", "targets", "gene"].edge_index = edge_index
            data["mirna", "targets", "gene"].edge_weight = edge_weight

        # Gene -> Gene edges (PPI, optional)
        if self.config["data"]["use_ppi_edges"] and "gene_gene" in self.edges_data:
            edge_index, edge_weight = self._build_edges(
                self.edges_data["gene_gene"], self.gene_to_idx, self.gene_to_idx
            )
            data["gene", "ppi", "gene"].edge_index = edge_index
            data["gene", "ppi", "gene"].edge_weight = edge_weight

        # Store patient ID
        data.patient_id = patient_id

        return data

    def _get_patient_features(self, df: pd.DataFrame, patient_id: str) -> torch.Tensor:
        """Extract features for a specific patient."""
        if patient_id not in df.columns:
            # Return zero features if patient not found
            return torch.zeros((df.shape[0],), dtype=torch.float32)

        features = df[patient_id].values

        # Handle mRNA and CNV concatenation if stored together
        # For now, keeping them separate as per instruction
        return torch.tensor(features, dtype=torch.float32)

    def _build_edges(
        self,
        edges_df: pd.DataFrame,
        source_mapping: Dict[str, int],
        target_mapping: Dict[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge index and weights from dataframe."""
        valid_edges = []
        weights = []

        for _, row in edges_df.iterrows():
            source = row["source"]
            target = row["target"]

            # Map IDs to indices
            if source in source_mapping and target in target_mapping:
                source_idx = source_mapping[source]
                target_idx = target_mapping[target]
                valid_edges.append([source_idx, target_idx])
                weights.append(row["weight"])

        if not valid_edges:
            # Return empty tensors if no valid edges
            return torch.tensor([[], []], dtype=torch.long), torch.tensor(
                [], dtype=torch.float32
            )

        edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
        edge_weight = torch.tensor(weights, dtype=torch.float32)

        return edge_index, edge_weight
