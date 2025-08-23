# src/losses/recon_feature.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FeatureReconstructionLoss(nn.Module):
    """MSE loss for feature reconstruction per node type."""

    def __init__(self, modality_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.modality_weights = modality_weights or {}

    def forward(
        self, pred_dict: Dict[str, torch.Tensor], target_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature reconstruction loss for each node type.

        Returns dictionary of losses per node type.
        """
        losses = {}

        for node_type in pred_dict.keys():
            pred = pred_dict[node_type]
            target = target_dict[node_type]

            # MSE loss
            loss = F.mse_loss(pred, target, reduction="mean")

            # Apply modality weighting if specified
            if node_type == "gene":
                # Split contribution for mRNA and CNV
                weight = (
                    self.modality_weights.get("mRNA", 1.0)
                    + self.modality_weights.get("CNV", 1.0)
                ) / 2
            elif node_type == "cpg":
                weight = self.modality_weights.get("DNAmeth", 1.0)
            elif node_type == "mirna":
                weight = self.modality_weights.get("miRNA", 1.0)
            else:
                weight = 1.0

            losses[f"feat_{node_type}"] = loss * weight

        return losses


# src/losses/recon_edge.py
class EdgeReconstructionLoss(nn.Module):
    """Binary cross-entropy loss for edge reconstruction with negative sampling."""

    def __init__(self, neg_sample_ratio: int = 5):
        super().__init__()
        self.neg_sample_ratio = neg_sample_ratio

    def forward(self, pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute balanced BCE loss for edge prediction.

        Args:
            pos_pred: Predictions for positive edges
            neg_pred: Predictions for negative edges

        Returns:
            Balanced BCE loss
        """
        # Positive edges
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred, torch.ones_like(pos_pred), reduction="mean"
        )

        # Negative edges
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_pred, torch.zeros_like(neg_pred), reduction="mean"
        )

        # Balanced loss
        return (pos_loss + neg_loss) / 2

    def sample_negative_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes_src: int,
        num_nodes_dst: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample negative edges for training.

        Returns:
            Negative edge indices (2, num_neg_edges)
        """
        num_pos_edges = edge_index.shape[1]
        num_neg_edges = num_pos_edges * self.neg_sample_ratio

        # Create set of existing edges for fast lookup
        edge_set = set(map(tuple, edge_index.t().cpu().numpy()))

        neg_edges = []
        while len(neg_edges) < num_neg_edges:
            # Random sample
            src = torch.randint(0, num_nodes_src, (num_neg_edges,))
            dst = torch.randint(0, num_nodes_dst, (num_neg_edges,))

            # Filter out existing edges
            for s, d in zip(src.tolist(), dst.tolist()):
                if (s, d) not in edge_set and len(neg_edges) < num_neg_edges:
                    neg_edges.append([s, d])

        return torch.tensor(neg_edges, device=device).t()


# src/losses/consistency.py
class ConsistencyLoss(nn.Module):
    """
    Consistency loss to keep fused embedding close to individual modality vectors.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, fused_embedding: torch.Tensor, modality_vectors: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute consistency loss between fused and individual modality embeddings.

        Args:
            fused_embedding: (batch_size, hidden_channels)
            modality_vectors: Dict of modality vectors

        Returns:
            Consistency loss
        """
        total_loss = 0
        num_modalities = len(modality_vectors)

        for modality_name, modality_vec in modality_vectors.items():
            # L2 distance between fused and modality vectors
            loss = F.mse_loss(fused_embedding, modality_vec, reduction="mean")
            total_loss += loss

        return total_loss / num_modalities


# src/losses/entropy_reg.py
class AttentionEntropyLoss(nn.Module):
    """
    Entropy regularizer to avoid peaked attention weights.
    """

    def __init__(self, target_entropy: Optional[float] = None):
        super().__init__()
        self.target_entropy = target_entropy

    def forward(self, attention_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute entropy regularization for attention weights.

        Args:
            attention_weights: Dictionary of attention weights per modality
                              Each tensor is (batch_size,)

        Returns:
            Negative entropy loss
        """
        # Stack attention weights: (batch_size, num_modalities)
        weights = torch.stack(list(attention_weights.values()), dim=1)

        # Compute entropy
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)

        if self.target_entropy is not None:
            # Penalize deviation from target entropy
            loss = F.mse_loss(entropy, torch.full_like(entropy, self.target_entropy))
        else:
            # Maximize entropy (minimize negative entropy)
            loss = -entropy.mean()

        return loss


# src/losses/weighting.py
def compute_modality_weights(node_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute per-modality weights based on node counts.
    Using inverse of node count to balance modalities.
    """
    # Map node types to modalities
    modality_counts = {
        "mRNA": node_counts["gene"],
        "CNV": node_counts["gene"],
        "DNAmeth": node_counts["cpg"],
        "miRNA": node_counts["mirna"],
    }

    # Inverse weighting
    total_nodes = sum(modality_counts.values())
    weights = {
        modality: total_nodes / (count * len(modality_counts))
        for modality, count in modality_counts.items()
    }

    # Normalize to sum to 1
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}

    return weights
