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

        # print(f"Gene embeddings shape: {node_embeddings['gene'].shape}")

        # Get batch assignments if batched
        batch_dict = None
        batch_dict = {
            "gene": data["gene"].batch,
            "cpg": data["cpg"].batch,
            "mirna": data["mirna"].batch,
        }

        # Pool to modality embeddings
        modality_embeddings = self.modality_pooling(node_embeddings, batch_dict)

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


class MultiModalGNNWithDecoders(nn.Module):
    """
    Complete model with decoders for reconstruction.
    """

    def __init__(self, config: dict):
        """
        Initialize model with decoders.

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        # Main model
        self.model = MultiModalHeteroGNN(config)

        # Import decoder modules
        from ..losses.recon_feature import MultiModalFeatureDecoders
        from ..losses.recon_edge import EdgeReconstructionLoss
        from ..losses.consistency import ConsistencyLoss, EntropyRegularization

        # Feature decoders
        self.feature_decoders = MultiModalFeatureDecoders(config)

        # Edge decoder
        self.edge_decoder = EdgeReconstructionLoss(
            neg_sampling_ratio=config["losses"]["neg_sampling_ratio"],
            decoder_type="inner_product",
            hidden_size=config["model"]["encoder"]["hidden_size"],
        )

        # Consistency loss
        self.consistency_loss = ConsistencyLoss(distance_metric="l2", normalize=True)

        # Entropy regularization
        self.entropy_reg = EntropyRegularization()

        self.config = config

    def forward(self, data: HeteroData, compute_loss: bool = False) -> Dict:
        """
        Forward pass with optional loss computation.

        Args:
            data: Batched HeteroData
            compute_loss: Whether to compute losses

        Returns:
            Dictionary containing model outputs and optionally losses
        """
        # Forward through main model
        output = self.model(data)

        if compute_loss:
            losses = {}

            # Feature reconstruction
            reconstructed = self.feature_decoders(output["node_embeddings"])

            # Prepare original features
            original = {
                "gene_mrna": data.gene_mrna_batched,
                "gene_cnv": data.gene_cnv_batched,
                "cpg": data["cpg"].x,
                "mirna": data["mirna"].x,
            }

            # Import loss module
            from ..losses.recon_feature import FeatureReconstructionLoss

            recon_loss_fn = FeatureReconstructionLoss(self.config)
            recon_loss, recon_losses_detail = recon_loss_fn(reconstructed, original)

            losses["recon_total"] = recon_loss
            losses["recon_detail"] = recon_losses_detail

            # Edge reconstruction
            edge_index_dict = {}
            for edge_type in self.model.edge_types:
                if hasattr(data[edge_type], "edge_index"):
                    edge_index_dict[edge_type] = data[edge_type].edge_index

            edge_loss, edge_losses_detail = self.edge_decoder(
                output["node_embeddings"], edge_index_dict
            )

            losses["edge_total"] = edge_loss * self.config["losses"]["lambda_edge"]
            losses["edge_detail"] = edge_losses_detail

            # Stage B specific losses
            if self.model.training_stage == "B":
                # Consistency loss
                cons_loss, cons_detail = self.consistency_loss(
                    output["fused_embedding"], output["modality_embeddings"]
                )
                losses["consistency"] = cons_loss * self.config["losses"]["lambda_cons"]
                losses["consistency_detail"] = cons_detail

                # Entropy regularization
                if output["attention_weights"] is not None:
                    ent_loss = self.entropy_reg(output["attention_weights"])
                    losses["entropy"] = ent_loss * self.config["losses"]["lambda_ent"]

            # Total loss
            total_loss = losses["recon_total"] + losses["edge_total"]

            if self.model.training_stage == "B":
                total_loss += losses.get("consistency", 0)
                total_loss += losses.get("entropy", 0)

            losses["total"] = total_loss

            output["losses"] = losses

        return output

    def set_training_stage(self, stage: str):
        """Set training stage."""
        self.model.set_training_stage(stage)
