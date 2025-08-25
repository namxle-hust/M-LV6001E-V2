"""
Edge reconstruction loss with negative sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from typing import Dict, Tuple, Optional
import numpy as np

class EdgeDecoder(nn.Module):
    """Decoder for edge prediction between node pairs."""

    def __init__(
        self, decoder_type: str = "inner_product", hidden_size: Optional[int] = None
    ):
        """
        Initialize edge decoder.

        Args:
            decoder_type: Type of decoder ('inner_product', 'mlp')
            hidden_size: Hidden size for MLP decoder
        """
        super().__init__()

        self.decoder_type = decoder_type

        if decoder_type == "mlp" and hidden_size is not None:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
            )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """
        Decode edge probability between source and destination nodes.

        Args:
            z_src: Source node embeddings [n_edges, hidden_size]
            z_dst: Destination node embeddings [n_edges, hidden_size]

        Returns:
            Edge probabilities [n_edges]
        """
        if self.decoder_type == "inner_product":
            # Inner product decoder
            return (z_src * z_dst).sum(dim=-1)

        elif self.decoder_type == "mlp":
            # MLP decoder
            concat = torch.cat([z_src, z_dst], dim=-1)
            return self.mlp(concat).squeeze(-1)

        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")


class EdgeReconstructionLoss(nn.Module):
    """
    Edge reconstruction loss with negative sampling.
    """

    def __init__(
        self,
        neg_sampling_ratio: int = 5,
        decoder_type: str = "inner_product",
        hidden_size: Optional[int] = None,
    ):
        """
        Initialize edge reconstruction loss.

        Args:
            neg_sampling_ratio: Ratio of negative to positive edges
            decoder_type: Type of edge decoder
            hidden_size: Hidden size for embeddings
        """
        super().__init__()

        self.neg_sampling_ratio = neg_sampling_ratio

        # Create decoders for each edge type
        self.decoders = nn.ModuleDict()

        # We'll initialize decoders dynamically based on edge types
        self.decoder_type = decoder_type
        self.hidden_size = hidden_size

    def get_decoder(self, edge_type: Tuple[str, str, str]) -> EdgeDecoder:
        """Get or create decoder for edge type."""
        edge_key = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"

        if edge_key not in self.decoders:
            self.decoders[edge_key] = EdgeDecoder(self.decoder_type, self.hidden_size)

        return self.decoders[edge_key]

    def sample_negative_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes_src: int,
        num_nodes_dst: int,
        num_neg_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample negative edges.

        Args:
            edge_index: Positive edges [2, n_edges]
            num_nodes_src: Number of source nodes
            num_nodes_dst: Number of destination nodes
            num_neg_samples: Number of negative samples

        Returns:
            Negative edge indices [2, n_neg_edges]
        """
        if num_neg_samples is None:
            num_neg_samples = edge_index.shape[1] * self.neg_sampling_ratio

        # For heterogeneous case, we need custom negative sampling
        neg_edges = []

        # Create set of existing edges for fast lookup
        edge_set = set()
        for i in range(edge_index.shape[1]):
            edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))

        # Sample negative edges
        while len(neg_edges) < num_neg_samples:
            src = torch.randint(0, num_nodes_src, (1,)).item()
            dst = torch.randint(0, num_nodes_dst, (1,)).item()

            if (src, dst) not in edge_set:
                neg_edges.append([src, dst])
                edge_set.add((src, dst))  # Avoid duplicates

        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()

        return neg_edge_index.to(edge_index.device)

    def forward(
        self,
        node_embeddings: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute edge reconstruction loss.

        Args:
            node_embeddings: Node embeddings by type
            edge_index_dict: Edge indices by relation type
            batch_dict: Batch assignments (for batched graphs)

        Returns:
            - Total edge loss
            - Individual losses by edge type
        """
        losses = {}

        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.shape[1] == 0:
                continue

            src_type, rel_type, dst_type = edge_type

            # Skip reverse edges to avoid double counting
            if rel_type in ["mapped_by", "targeted_by"]:
                continue

            # Get embeddings
            z_src = node_embeddings[src_type]
            z_dst = node_embeddings[dst_type]

            # Get decoder for this edge type
            decoder = self.get_decoder(edge_type)

            # Positive edges
            pos_src = z_src[edge_index[0]]
            pos_dst = z_dst[edge_index[1]]
            pos_pred = decoder(pos_src, pos_dst)

            # Sample negative edges
            num_nodes_src = z_src.shape[0]
            num_nodes_dst = z_dst.shape[0]

            neg_edge_index = self.sample_negative_edges(
                edge_index, num_nodes_src, num_nodes_dst
            )

            # Negative edges
            neg_src = z_src[neg_edge_index[0]]
            neg_dst = z_dst[neg_edge_index[1]]
            neg_pred = decoder(neg_src, neg_dst)

            # Binary cross-entropy loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_pred, torch.ones_like(pos_pred)
            )

            neg_loss = F.binary_cross_entropy_with_logits(
                neg_pred, torch.zeros_like(neg_pred)
            )

            # Balanced loss
            edge_loss = (pos_loss + neg_loss) / 2

            losses[f"{src_type}_{rel_type}_{dst_type}"] = edge_loss

        # Total loss
        if losses:
            total_loss = sum(losses.values()) / len(losses)
        else:
            total_loss = torch.tensor(
                0.0, device=node_embeddings[list(node_embeddings.keys())[0]].device
            )

        return total_loss, losses


class LinkPredictionMetrics:
    """Compute link prediction metrics."""

    @staticmethod
    def compute_auc_ap(
        pos_pred: torch.Tensor, neg_pred: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute AUC and Average Precision.

        Args:
            pos_pred: Positive edge predictions
            neg_pred: Negative edge predictions

        Returns:
            Dictionary with 'auroc' and 'auprc'
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Combine predictions and labels
        preds = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        labels = (
            torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
            .cpu()
            .numpy()
        )

        # Compute metrics
        auroc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)

        return {"auroc": auroc, "auprc": auprc}
