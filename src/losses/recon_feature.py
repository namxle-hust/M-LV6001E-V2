"""
Feature reconstruction loss for node embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FeatureDecoder(nn.Module):
    """Decoder for reconstructing node features from embeddings."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        hidden_layers: list = [128, 64],
        dropout: float = 0.1,
    ):
        """
        Initialize feature decoder.

        Args:
            hidden_size: Input embedding size
            output_size: Output feature size
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        in_size = hidden_size

        for out_size in hidden_layers:
            layers.extend(
                [
                    nn.Linear(in_size, out_size),
                    nn.LayerNorm(out_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_size = out_size

        # Final projection
        layers.append(nn.Linear(in_size, output_size))

        self.decoder = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to features.

        Args:
            embeddings: Node embeddings [n_nodes, hidden_size]

        Returns:
            Reconstructed features [n_nodes, output_size]
        """
        return self.decoder(embeddings)


class MultiModalFeatureDecoders(nn.Module):
    """Multiple decoders for different feature types."""

    def __init__(self, config: dict):
        """
        Initialize multi-modal decoders.

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        hidden_size = config["model"]["encoder"]["hidden_size"]
        decoder_config = config["model"]["decoders"]

        # Gene decoders (separate for mrna and cnv)
        self.mrna_decoder = FeatureDecoder(
            hidden_size=hidden_size,
            output_size=1,  # Single channel for mrna
            hidden_layers=decoder_config["gene_decoder"]["hidden_sizes"],
            dropout=decoder_config["gene_decoder"]["dropout"],
        )

        self.cnv_decoder = FeatureDecoder(
            hidden_size=hidden_size,
            output_size=1,  # Single channel for cnv
            hidden_layers=decoder_config["gene_decoder"]["hidden_sizes"],
            dropout=decoder_config["gene_decoder"]["dropout"],
        )

        # cpg decoder
        self.cpg_decoder = FeatureDecoder(
            hidden_size=hidden_size,
            output_size=1,
            hidden_layers=decoder_config["cpg_decoder"]["hidden_sizes"],
            dropout=decoder_config["cpg_decoder"]["dropout"],
        )

        # mirna decoder
        self.mirna_decoder = FeatureDecoder(
            hidden_size=hidden_size,
            output_size=1,
            hidden_layers=decoder_config["mirna_decoder"]["hidden_sizes"],
            dropout=decoder_config["mirna_decoder"]["dropout"],
        )

    def forward(
        self, node_embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Decode all node embeddings.

        Args:
            node_embeddings: Node embeddings by type

        Returns:
            Reconstructed features by type
        """
        reconstructed = {}

        # Decode gene embeddings to mrna and cnv separately
        gene_emb = node_embeddings["gene"]
        reconstructed["gene_mrna"] = self.mrna_decoder(gene_emb)
        reconstructed["gene_cnv"] = self.cnv_decoder(gene_emb)

        # Decode other node types
        reconstructed["cpg"] = self.cpg_decoder(node_embeddings["cpg"])
        reconstructed["mirna"] = self.mirna_decoder(node_embeddings["mirna"])

        return reconstructed


class FeatureReconstructionLoss(nn.Module):
    """
    Compute feature reconstruction loss with per-modality weighting.
    """

    def __init__(self, config: dict):
        """
        Initialize reconstruction loss.

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.loss_weights = {
            "mrna": config["losses"]["lambda_recon_mrna"],
            "cnv": config["losses"]["lambda_recon_cnv"],
            "cpg": config["losses"]["lambda_recon_cpg"],
            "mirna": config["losses"]["lambda_recon_mirna"],
        }

        # Modality weights for balancing
        self.modality_weights = config["losses"].get("modality_weights", {})

    def compute_modality_weights(self, num_nodes: Dict[str, int]) -> Dict[str, float]:
        """
        Compute modality weights based on node counts.

        Args:
            num_nodes: Number of nodes per type

        Returns:
            Normalized modality weights
        """
        if all(w is not None for w in self.modality_weights.values()):
            return self.modality_weights

        # Auto-compute weights based on inverse node count
        weights = {}
        total_nodes = sum(num_nodes.values())

        for node_type, count in num_nodes.items():
            weights[node_type] = total_nodes / (count + 1e-8)

        # Normalize
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight

        return weights

    def forward(
        self,
        reconstructed: Dict[str, torch.Tensor],
        original: Dict[str, torch.Tensor],
        num_nodes: Optional[Dict[str, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reconstruction loss.

        Args:
            reconstructed: Reconstructed features
            original: Original features
            num_nodes: Number of nodes per type

        Returns:
            - Total reconstruction loss
            - Individual losses by type
        """
        losses = {}

        # Gene mrna reconstruction
        if "gene_mrna" in reconstructed and "gene_mrna" in original:
            # Flatten original to match decoder output: [n_patients, n_genes] -> [n_patients * n_genes]
            original_mrna_flat = original["gene_mrna"].view(-1)
            mrna_loss = F.mse_loss(
                reconstructed["gene_mrna"].squeeze(), original_mrna_flat
            )
            losses["mrna"] = mrna_loss * self.loss_weights["mrna"]

        # Gene cnv reconstruction
        if "gene_cnv" in reconstructed and "gene_cnv" in original:
            # Flatten original to match decoder output: [n_patients, n_genes] -> [n_patients * n_genes]
            original_cnv_flat = original["gene_cnv"].view(-1)
            cnv_loss = F.mse_loss(
                reconstructed["gene_cnv"].squeeze(), original_cnv_flat
            )
            losses["cnv"] = cnv_loss * self.loss_weights["cnv"]

        # cpg reconstruction
        if "cpg" in reconstructed and "cpg" in original:
            cpg_loss = F.mse_loss(
                reconstructed["cpg"].squeeze(), original["cpg"].squeeze()
            )
            losses["cpg"] = cpg_loss * self.loss_weights["cpg"]

        # mirna reconstruction
        if "mirna" in reconstructed and "mirna" in original:
            mirna_loss = F.mse_loss(
                reconstructed["mirna"].squeeze(), original["mirna"].squeeze()
            )
            losses["mirna"] = mirna_loss * self.loss_weights["mirna"]

        # Apply modality weights if provided
        if num_nodes is not None:
            modality_weights = self.compute_modality_weights(num_nodes)

            weighted_losses = {}
            for key, loss in losses.items():
                weight_key = key if key in modality_weights else "gene"
                weighted_losses[key] = loss * modality_weights.get(weight_key, 1.0)

            total_loss = sum(weighted_losses.values())
        else:
            total_loss = sum(losses.values())

        return total_loss, losses


class ContrastiveLoss(nn.Module):
    """
    Optional contrastive regularization for robustness.
    """

    def __init__(self, temperature: float = 0.1, noise_std: float = 0.1):
        """
        Initialize contrastive loss.

        Args:
            temperature: Temperature for similarity
            noise_std: Standard deviation for noise injection
        """
        super().__init__()

        self.temperature = temperature
        self.noise_std = noise_std

    def add_noise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to embeddings."""
        noise = torch.randn_like(embeddings) * self.noise_std
        return embeddings + noise

    def forward(self, embeddings: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Node embeddings
            batch_size: Batch size for negative sampling

        Returns:
            Contrastive loss value
        """
        # Create positive pairs by adding noise
        embeddings_aug = self.add_noise(embeddings)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        embeddings_aug = F.normalize(embeddings_aug, dim=-1)

        # Compute similarities
        sim_matrix = torch.matmul(embeddings, embeddings_aug.t()) / self.temperature

        # Positive pairs are on the diagonal
        labels = torch.arange(embeddings.size(0), device=embeddings.device)

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
