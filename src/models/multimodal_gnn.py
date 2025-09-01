"""
Main multi-modal heterogeneous GNN model.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional

from .hetero_encoder import create_encoder
from .modality_pool import ModalityPooling
from .modality_attention import ModalityAttention


class MultiModalHeteroGNN(nn.Module):
    """
    Complete multi-modal heterogeneous GNN for cancer genomics.
    """

    def __init__(self, config: dict):
        """
        Initialize model.

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.config = config

        # Define input channels for each node type
        in_channels_dict = {
            "gene": 2,  # mrna + cnv channels
            "cpg": 1,  # methylation values
            "mirna": 1,  # mirna expression
        }

        # Define edge types
        self.edge_types = [
            ("cpg", "maps_to", "gene"),
            ("mirna", "targets", "gene"),
            ("gene", "mapped_by", "cpg"),
            ("gene", "targeted_by", "mirna"),
        ]

        if config["data"].get("use_ppi", False):
            self.edge_types.append(("gene", "ppi", "gene"))

        # Create encoder
        self.encoder = create_encoder(config, in_channels_dict, self.edge_types)

        # Modality pooling
        self.modality_pooling = ModalityPooling(
            hidden_size=config["model"]["encoder"]["hidden_size"],
            pool_type=config["model"]["pooling"]["type"],
            use_projection_heads=True,
        )

        # Modality attention
        self.modality_attention = ModalityAttention(
            hidden_size=config["model"]["encoder"]["hidden_size"],
            attention_hidden=config["model"]["attention"]["hidden_size"],
            temperature=config["model"]["attention"]["temperature"],
            dropout=config["model"]["attention"]["dropout"],
        )

        self.training_stage = "A"  # Start with stage A

    def set_training_stage(self, stage: str):
        """Set training stage ('A' for pretrain, 'B' for fusion)."""
        self.training_stage = stage
        print(f"Training stage set to: {stage}")

    def forward(self, data: HeteroData) -> Dict:
        """
        Forward pass through the model.

        FIXED: Properly detect and use batch assignments.

        Args:
            data: Batched HeteroData

        Returns:
            Dictionary containing:
                - node_embeddings: Updated node embeddings
                - modality_embeddings: Modality-level embeddings
                - fused_embedding: Fused patient embedding
                - attention_weights: Attention weights (if stage B)
        """
        # Extract features and edges
        x_dict = {
            "gene": data["gene"].x,
            "cpg": data["cpg"].x,
            "mirna": data["mirna"].x,
        }

        edge_index_dict = {}
        for edge_type in self.edge_types:
            if hasattr(data[edge_type], "edge_index"):
                edge_index_dict[edge_type] = data[edge_type].edge_index

        # Encode nodes
        node_embeddings = self.encoder(x_dict, edge_index_dict)

        # FIX: Properly detect batch assignments
        # In PyTorch Geometric, batch assignments are stored per node type
        batch_dict = None

        # Check if this is a batched graph by looking for batch assignments in node data
        if hasattr(data["gene"], "batch"):
            batch_dict = {
                "gene": data["gene"].batch,
                "cpg": data["cpg"].batch,
                "mirna": data["mirna"].batch,
            }

            # Debug: Print batch information
            unique_batches = torch.unique(data["gene"].batch)
            num_patients = len(unique_batches)
            print(f"DEBUG: Detected {num_patients} patients in batch")
            print(f"DEBUG: Unique batch IDs: {unique_batches.tolist()}")

        # Pool to modality embeddings
        modality_embeddings = self.modality_pooling(node_embeddings, batch_dict)

        # Debug: Print modality embedding shapes
        for modality, emb in modality_embeddings.items():
            print(f"DEBUG: {modality} embedding shape: {emb.shape}")

        # Initialize output
        output = {
            "node_embeddings": node_embeddings,
            "modality_embeddings": modality_embeddings,
        }

        # Stage B: Apply attention fusion
        if self.training_stage == "B":
            fused_embedding, attention_weights, attention_dict = (
                self.modality_attention(modality_embeddings)
            )

            # Debug: Print attention weights shape
            print(f"DEBUG: Attention weights shape: {attention_weights.shape}")

            output["fused_embedding"] = fused_embedding
            output["attention_weights"] = attention_weights
            output["attention_dict"] = attention_dict
        else:
            # Stage A: Simple averaging for fused embedding
            embeddings_list = list(modality_embeddings.values())
            fused_embedding = torch.stack(embeddings_list, dim=0).mean(dim=0)
            output["fused_embedding"] = fused_embedding
            output["attention_weights"] = None
            output["attention_dict"] = None

        return output
