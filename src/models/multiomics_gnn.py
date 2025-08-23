# src/models/multiomics_gnn.py
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional


class MultiOmicsGNN(nn.Module):
    """
    Complete Multi-Omics Graph Neural Network model.
    Combines encoder, pooling, attention fusion, and decoders.
    """

    def __init__(self, config: dict, num_features_dict: Dict[str, int]):
        super().__init__()

        self.config = config
        self.num_features_dict = num_features_dict

        # Extract model config
        model_config = config["model"]
        hidden_dim = model_config["hidden_dim"]

        # Heterogeneous encoder
        self.encoder = HeteroGATEncoder(
            in_channels_dict=num_features_dict,
            hidden_channels=hidden_dim,
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            dropout=model_config["dropout"],
            layer_norm=model_config["layer_norm"],
            concat_heads=model_config["concat_heads"],
        )

        # Modality pooling
        self.modality_pool = ModalityPooling(
            hidden_channels=hidden_dim,
            pooling_type=model_config["pooling_type"],
            separate_gene_channels=False,
        )

        # Modality attention fusion
        self.modality_attention = ModalityAttention(
            hidden_channels=hidden_dim,
            attention_hidden=model_config["attention_hidden"],
            dropout=model_config["attention_dropout"],
        )

        # Feature decoder
        self.feature_decoder = FeatureDecoder(
            hidden_channels=hidden_dim, out_channels_dict=num_features_dict
        )

        # Edge decoder
        self.edge_decoder = EdgeDecoder(hidden_channels=hidden_dim, use_mlp=True)

        # Loss functions
        loss_config = config["losses"]
        self.feature_recon_loss = FeatureReconstructionLoss(
            modality_weights=loss_config.get("modality_weights")
        )
        self.edge_recon_loss = EdgeReconstructionLoss(
            neg_sample_ratio=config["training"]["neg_sample_ratio"]
        )
        self.consistency_loss = ConsistencyLoss()
        self.entropy_loss = AttentionEntropyLoss()

        # Loss weights
        self.loss_weights = {
            "feature": loss_config["feature_recon"],
            "edge": loss_config["edge_recon"],
            "consistency": loss_config["consistency"],
            "entropy": loss_config["entropy_reg"],
        }

    def forward(self, data: HeteroData, return_attention: bool = False) -> Dict:
        """
        Forward pass through the model.

        Args:
            data: Heterogeneous graph data
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - node_embeddings: Updated node embeddings
                - modality_vectors: Modality-level representations
                - patient_embedding: Fused patient embedding
                - attention_weights: Attention weights (if requested)
                - reconstructed_features: Reconstructed node features
        """
        # Extract data
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Encode nodes
        h_dict = self.encoder(x_dict, edge_index_dict)

        # Pool to modality vectors
        batch_dict = data.batch_dict if hasattr(data, "batch_dict") else None
        modality_vectors = self.modality_pool(h_dict, batch_dict)

        # Attention fusion
        patient_embedding, attention_weights = self.modality_attention(modality_vectors)

        # Decode features
        reconstructed_features = self.feature_decoder(h_dict)

        results = {
            "node_embeddings": h_dict,
            "modality_vectors": modality_vectors,
            "patient_embedding": patient_embedding,
            "reconstructed_features": reconstructed_features,
        }

        if return_attention:
            results["attention_weights"] = attention_weights

        return results

    def compute_loss(
        self, results: Dict, data: HeteroData, stage: str = "full"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss based on training stage.

        Args:
            results: Model outputs
            data: Original data
            stage: 'pretrain' or 'full'

        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses for logging
        """
        loss_dict = {}
        total_loss = 0

        # Feature reconstruction loss
        feat_losses = self.feature_recon_loss(
            results["reconstructed_features"], data.x_dict
        )
        for key, loss in feat_losses.items():
            loss_dict[key] = loss
            total_loss += loss * self.loss_weights["feature"]

        # Edge reconstruction loss
        for edge_type, edge_index in data.edge_index_dict.items():
            if edge_index.shape[1] == 0:
                continue

            src_type, _, dst_type = edge_type

            # Get embeddings for edge endpoints
            src_embeddings = results["node_embeddings"][src_type][edge_index[0]]
            dst_embeddings = results["node_embeddings"][dst_type][edge_index[1]]

            # Positive edge predictions
            edge_type_str = f"{src_type}_{dst_type}"
            pos_pred = self.edge_decoder(src_embeddings, dst_embeddings, edge_type_str)

            # Sample negative edges
            neg_edge_index = self.edge_recon_loss.sample_negative_edges(
                edge_index,
                data[src_type].num_nodes,
                data[dst_type].num_nodes,
                edge_index.device,
            )

            # Negative edge predictions
            neg_src_embeddings = results["node_embeddings"][src_type][neg_edge_index[0]]
            neg_dst_embeddings = results["node_embeddings"][dst_type][neg_edge_index[1]]
            neg_pred = self.edge_decoder(
                neg_src_embeddings, neg_dst_embeddings, edge_type_str
            )

            # Compute loss
            edge_loss = self.edge_recon_loss(pos_pred, neg_pred)
            loss_dict[f"edge_{edge_type_str}"] = edge_loss
            total_loss += edge_loss * self.loss_weights["edge"]

        # Stage-specific losses
        if stage == "full":
            # Consistency loss
            cons_loss = self.consistency_loss(
                results["patient_embedding"], results["modality_vectors"]
            )
            loss_dict["consistency"] = cons_loss
            total_loss += cons_loss * self.loss_weights["consistency"]

            # Entropy regularization
            if "attention_weights" in results:
                ent_loss = self.entropy_loss(results["attention_weights"])
                loss_dict["entropy"] = ent_loss
                total_loss += ent_loss * self.loss_weights["entropy"]

        loss_dict["total"] = total_loss

        return total_loss, loss_dict

    def get_patient_embedding(self, data: HeteroData) -> torch.Tensor:
        """Get final patient embedding."""
        results = self.forward(data)
        return results["patient_embedding"]

    def get_modality_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Get modality-specific embeddings."""
        results = self.forward(data)
        return results["modality_vectors"]
