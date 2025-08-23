# src/models/modality_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ModalityAttention(nn.Module):
    """
    Attention mechanism to fuse modality-level representations.
    Computes attention weights for each modality and creates fused patient embedding.
    """

    def __init__(
        self, hidden_channels: int, attention_hidden: int = 128, dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.attention_hidden = attention_hidden

        # Shared MLP for scoring modalities
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_channels, attention_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden, 1),
        )

    def forward(
        self, modality_vectors: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute attention-weighted fusion of modality vectors.

        Args:
            modality_vectors: Dictionary with keys ['mRNA', 'CNV', 'DNAmeth', 'miRNA']
                             Each value is shape (batch_size, hidden_channels)

        Returns:
            fused_embedding: (batch_size, hidden_channels) - Fused patient embedding
            attention_weights: Dictionary of attention weights for each modality
        """

        # Stack modality vectors
        modality_names = ["mRNA", "CNV", "DNAmeth", "miRNA"]
        modality_tensors = [modality_vectors[name] for name in modality_names]

        # Shape: (batch_size, num_modalities, hidden_channels)
        stacked_modalities = torch.stack(modality_tensors, dim=1)
        batch_size = stacked_modalities.shape[0]

        # Compute attention scores
        # Reshape for MLP: (batch_size * num_modalities, hidden_channels)
        reshaped = stacked_modalities.view(-1, self.hidden_channels)
        scores = self.attention_mlp(reshaped)  # (batch_size * num_modalities, 1)
        scores = scores.view(
            batch_size, len(modality_names)
        )  # (batch_size, num_modalities)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, num_modalities)

        # Weighted sum of modality vectors
        # Expand attention weights for broadcasting
        attention_expanded = attention_weights.unsqueeze(
            -1
        )  # (batch_size, num_modalities, 1)

        # Compute weighted sum
        fused_embedding = (stacked_modalities * attention_expanded).sum(
            dim=1
        )  # (batch_size, hidden_channels)

        # Create attention weight dictionary for interpretability
        attention_dict = {
            name: attention_weights[:, i] for i, name in enumerate(modality_names)
        }

        return fused_embedding, attention_dict


# src/models/decoders.py
class FeatureDecoder(nn.Module):
    """Decoder for reconstructing node features."""

    def __init__(self, hidden_channels: int, out_channels_dict: Dict[str, int]):
        super().__init__()

        self.decoders = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_channels // 2, out_channels),
                )
                for node_type, out_channels in out_channels_dict.items()
            }
        )

    def forward(self, h_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode node embeddings back to features."""
        return {
            node_type: self.decoders[node_type](h) for node_type, h in h_dict.items()
        }


class EdgeDecoder(nn.Module):
    """Decoder for edge reconstruction using inner product or MLP."""

    def __init__(self, hidden_channels: int, use_mlp: bool = False):
        super().__init__()

        self.use_mlp = use_mlp

        if use_mlp:
            self.edge_mlps = nn.ModuleDict(
                {
                    "cpg_gene": nn.Sequential(
                        nn.Linear(hidden_channels * 2, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, 1),
                    ),
                    "mirna_gene": nn.Sequential(
                        nn.Linear(hidden_channels * 2, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, 1),
                    ),
                    "gene_gene": nn.Sequential(
                        nn.Linear(hidden_channels * 2, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, 1),
                    ),
                }
            )

    def forward(
        self, z_src: torch.Tensor, z_dst: torch.Tensor, edge_type: str
    ) -> torch.Tensor:
        """
        Predict edge probability between source and destination nodes.

        Args:
            z_src: Source node embeddings (num_edges, hidden_channels)
            z_dst: Destination node embeddings (num_edges, hidden_channels)
            edge_type: Type of edge being decoded

        Returns:
            Edge predictions (num_edges,)
        """
        if self.use_mlp:
            z_concat = torch.cat([z_src, z_dst], dim=-1)
            return self.edge_mlps[edge_type](z_concat).squeeze(-1)
        else:
            # Simple inner product
            return (z_src * z_dst).sum(dim=-1)
