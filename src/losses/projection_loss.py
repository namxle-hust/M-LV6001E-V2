"""
Projection training loss for Stage A.

This loss ensures that projection heads (mrna_projection, cnv_projection) learn
meaningful representations during Stage A, rather than remaining randomly initialized.

Strategy: Ensure that pooled modality embeddings preserve enough information
to reconstruct the original features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ProjectionReconstructionLoss(nn.Module):
    """
    Loss that trains projection heads by ensuring pooled embeddings
    can still reconstruct original features.

    This bridges the gap between:
    - Node-level reconstruction (direct path: gene_emb → decoder → features)
    - Modality-level pooling (projection path: gene_emb → projection → pool → z_modality)

    By adding this loss in Stage A, we ensure projections learn to extract
    reconstruction-relevant features, making Stage B more stable.
    """

    def __init__(
        self,
        config: dict,
        decoder_type: str = "simple"
    ):
        """
        Initialize projection reconstruction loss.

        Args:
            config: Configuration dictionary
            decoder_type: Type of decoder to use ('simple' or 'shared')
                - 'simple': Small MLP to decode pooled embeddings
                - 'shared': Reuse existing feature decoders (requires unpooling)
        """
        super().__init__()

        self.decoder_type = decoder_type
        hidden_size = config["model"]["encoder"]["hidden_size"]

        # Create simple decoders for pooled embeddings
        # These decode patient-level embeddings back to patient-level features
        if decoder_type == "simple":
            # For genes: predict average mRNA/CNV per patient
            self.pooled_mrna_decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)  # Predict mean mRNA value
            )

            self.pooled_cnv_decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)  # Predict mean CNV value
            )

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        original_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute projection reconstruction loss.

        Args:
            modality_embeddings: Pooled modality embeddings
                - 'mrna': [batch_size, hidden_size]
                - 'cnv': [batch_size, hidden_size]
            original_features: Original features
                - 'gene_mrna': [batch_size, n_genes] or [n_genes_total]
                - 'gene_cnv': [batch_size, n_genes] or [n_genes_total]

        Returns:
            - Total projection loss
            - Individual losses by modality
        """
        losses = {}

        # mRNA projection loss
        if 'mrna' in modality_embeddings and 'gene_mrna' in original_features:
            z_mrna = modality_embeddings['mrna']  # [batch_size, hidden_size]

            # Decode to predict mean mRNA value per patient
            pred_mrna_mean = self.pooled_mrna_decoder(z_mrna).squeeze(-1)  # [batch_size]

            # Compute actual mean mRNA per patient
            if original_features['gene_mrna'].dim() == 2:
                # Shape: [batch_size, n_genes]
                target_mrna_mean = original_features['gene_mrna'].mean(dim=1)  # [batch_size]
            else:
                # Shape: [n_genes_total] - need to compute per-patient means
                # This shouldn't happen with batched data, but handle it gracefully
                target_mrna_mean = original_features['gene_mrna'].mean()
                target_mrna_mean = target_mrna_mean.expand(z_mrna.size(0))

            # MSE loss
            losses['mrna'] = F.mse_loss(pred_mrna_mean, target_mrna_mean)

        # CNV projection loss
        if 'cnv' in modality_embeddings and 'gene_cnv' in original_features:
            z_cnv = modality_embeddings['cnv']  # [batch_size, hidden_size]

            # Decode to predict mean CNV value per patient
            pred_cnv_mean = self.pooled_cnv_decoder(z_cnv).squeeze(-1)  # [batch_size]

            # Compute actual mean CNV per patient
            if original_features['gene_cnv'].dim() == 2:
                # Shape: [batch_size, n_genes]
                target_cnv_mean = original_features['gene_cnv'].mean(dim=1)  # [batch_size]
            else:
                # Shape: [n_genes_total]
                target_cnv_mean = original_features['gene_cnv'].mean()
                target_cnv_mean = target_cnv_mean.expand(z_cnv.size(0))

            # MSE loss
            losses['cnv'] = F.mse_loss(pred_cnv_mean, target_cnv_mean)

        # Total loss
        if losses:
            total_loss = sum(losses.values()) / len(losses)
        else:
            # No losses computed (shouldn't happen)
            total_loss = torch.tensor(0.0, device=modality_embeddings['mrna'].device)

        return total_loss, losses


class ProjectionAlignmentLoss(nn.Module):
    """
    Alternative: Alignment loss between node-level and pooled representations.

    Ensures that:
    1. Projected+pooled embeddings align with mean of projected node embeddings
    2. Projections extract consistent features at node and patient levels
    """

    def __init__(self, distance_metric: str = "cosine"):
        """
        Initialize projection alignment loss.

        Args:
            distance_metric: Distance metric ('cosine', 'l2', 'l1')
        """
        super().__init__()
        self.distance_metric = distance_metric

    def compute_distance(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between embeddings."""
        if self.distance_metric == "cosine":
            # Cosine similarity (higher is better, so use 1 - sim as loss)
            cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)
            return (1 - cos_sim).mean()
        elif self.distance_metric == "l2":
            return F.mse_loss(emb1, emb2)
        elif self.distance_metric == "l1":
            return F.l1_loss(emb1, emb2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def forward(
        self,
        projected_node_embeddings: Dict[str, torch.Tensor],
        pooled_modality_embeddings: Dict[str, torch.Tensor],
        batch_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute alignment loss.

        Args:
            projected_node_embeddings: Node-level projected embeddings
                - 'mrna': [n_genes_total, hidden_size] (after mrna_projection)
                - 'cnv': [n_genes_total, hidden_size] (after cnv_projection)
            pooled_modality_embeddings: Patient-level pooled embeddings
                - 'mrna': [batch_size, hidden_size]
                - 'cnv': [batch_size, hidden_size]
            batch_dict: Batch assignments for nodes
                - 'gene': [n_genes_total] with values [0, 1, ..., batch_size-1]

        Returns:
            - Total alignment loss
            - Individual losses by modality
        """
        from torch_geometric.nn import global_mean_pool

        losses = {}

        # For mRNA
        if 'mrna' in projected_node_embeddings and 'mrna' in pooled_modality_embeddings:
            # Manually pool the node-level projections
            node_mrna = projected_node_embeddings['mrna']  # [n_genes_total, hidden]
            pooled_from_nodes = global_mean_pool(node_mrna, batch_dict['gene'])  # [batch, hidden]

            # Compare with the actual pooled embeddings
            pooled_mrna = pooled_modality_embeddings['mrna']  # [batch, hidden]

            losses['mrna'] = self.compute_distance(pooled_from_nodes, pooled_mrna)

        # For CNV
        if 'cnv' in projected_node_embeddings and 'cnv' in pooled_modality_embeddings:
            node_cnv = projected_node_embeddings['cnv']
            pooled_from_nodes = global_mean_pool(node_cnv, batch_dict['gene'])
            pooled_cnv = pooled_modality_embeddings['cnv']

            losses['cnv'] = self.compute_distance(pooled_from_nodes, pooled_cnv)

        # Total loss
        if losses:
            total_loss = sum(losses.values()) / len(losses)
        else:
            total_loss = torch.tensor(0.0)

        return total_loss, losses


class ProjectionRegularizationLoss(nn.Module):
    """
    Regularization loss to prevent projection heads from collapsing or diverging.

    Encourages:
    1. Projections to stay close to identity initially (smooth learning)
    2. Different projections to produce different outputs (diversity)
    """

    def __init__(self, reg_type: str = "orthogonal", weight: float = 0.01):
        """
        Initialize projection regularization.

        Args:
            reg_type: Type of regularization
                - 'identity': Encourage W ≈ I
                - 'orthogonal': Encourage W_mrna ⊥ W_cnv
                - 'diversity': Encourage z_mrna ≠ z_cnv
            weight: Regularization weight
        """
        super().__init__()
        self.reg_type = reg_type
        self.weight = weight

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        projection_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute regularization loss.

        Args:
            modality_embeddings: Pooled modality embeddings
            projection_weights: Projection weight matrices (optional)

        Returns:
            Regularization loss
        """
        if self.reg_type == "diversity":
            # Encourage different modalities to have different embeddings
            # Compute cosine similarity between mrna and cnv embeddings
            z_mrna = modality_embeddings['mrna']  # [batch, hidden]
            z_cnv = modality_embeddings['cnv']    # [batch, hidden]

            # We want them to be different, so penalize high similarity
            cos_sim = F.cosine_similarity(z_mrna, z_cnv, dim=-1)

            # Loss is high when similarity is high
            diversity_loss = cos_sim.abs().mean()

            return diversity_loss * self.weight

        elif self.reg_type == "orthogonal" and projection_weights:
            # Encourage projection matrices to be orthogonal
            W_mrna = projection_weights['mrna']  # [hidden, hidden]
            W_cnv = projection_weights['cnv']    # [hidden, hidden]

            # Compute W_mrna^T · W_cnv (should be close to 0 if orthogonal)
            product = torch.matmul(W_mrna.t(), W_cnv)

            # Frobenius norm of the product
            orthogonal_loss = torch.norm(product, p='fro')

            return orthogonal_loss * self.weight

        else:
            return torch.tensor(0.0)
