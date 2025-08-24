"""
Modality-level attention fusion module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ModalityAttention(nn.Module):
    """
    Attention mechanism to fuse multiple modality embeddings.
    """

    def __init__(
        self,
        hidden_size: int,
        attention_hidden: int = 128,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        """
        Initialize modality attention module.

        Args:
            hidden_size: Size of modality embeddings
            attention_hidden: Hidden size for attention MLP
            temperature: Temperature for softmax
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.temperature = temperature

        # Shared MLP for computing attention scores
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden),
            nn.LayerNorm(attention_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden, 1),
        )

        # Optional: Modality-specific transformations before attention
        self.modality_keys = ["mRNA", "CNV", "DNAmeth", "miRNA"]

    def compute_attention_weights(
        self, modality_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention weights for each modality.

        Args:
            modality_embeddings: Dictionary of modality embeddings

        Returns:
            Attention weights [batch_size, num_modalities]
        """
        scores = []

        for modality in self.modality_keys:
            if modality in modality_embeddings:
                emb = modality_embeddings[modality]
                score = self.attention_mlp(emb)  # [batch_size, 1]
                scores.append(score)

        # Stack scores
        scores = torch.cat(scores, dim=1)  # [batch_size, num_modalities]

        # Apply temperature scaling and softmax
        attention_weights = F.softmax(scores / self.temperature, dim=1)

        return attention_weights

    def forward(
        self, modality_embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Fuse modality embeddings using attention.

        Args:
            modality_embeddings: Dictionary of modality embeddings

        Returns:
            - Fused patient embedding
            - Attention weights
            - Dictionary of attention scores by modality
        """
        # Compute attention weights
        attention_weights = self.compute_attention_weights(modality_embeddings)

        # Stack modality embeddings
        embeddings_list = []
        present_modalities = []

        for modality in self.modality_keys:
            if modality in modality_embeddings:
                embeddings_list.append(modality_embeddings[modality])
                present_modalities.append(modality)

        stacked_embeddings = torch.stack(
            embeddings_list, dim=1
        )  # [batch_size, num_modalities, hidden_size]

        # Apply attention weights
        attention_weights_expanded = attention_weights.unsqueeze(
            -1
        )  # [batch_size, num_modalities, 1]
        fused_embedding = (stacked_embeddings * attention_weights_expanded).sum(
            dim=1
        )  # [batch_size, hidden_size]

        # Create attention score dictionary
        attention_dict = {}
        for i, modality in enumerate(present_modalities):
            attention_dict[modality] = attention_weights[:, i].mean().item()

        return fused_embedding, attention_weights, attention_dict


class CrossModalityAttention(nn.Module):
    """
    Cross-modality attention for learning modality interactions.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize cross-modality attention.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert (
            hidden_size % num_heads == 0
        ), "Hidden size must be divisible by num_heads"

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, modality_embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply cross-modality attention.

        Args:
            modality_embeddings: Dictionary of modality embeddings

        Returns:
            Updated modality embeddings with cross-modality information
        """
        # Stack modality embeddings
        modality_list = list(modality_embeddings.keys())
        embeddings = torch.stack([modality_embeddings[m] for m in modality_list], dim=1)
        batch_size, num_modalities, hidden_size = embeddings.shape

        # Compute Q, K, V
        Q = (
            self.q_linear(embeddings)
            .view(batch_size, num_modalities, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [batch, heads, modalities, head_dim]

        K = (
            self.k_linear(embeddings)
            .view(batch_size, num_modalities, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        V = (
            self.v_linear(embeddings)
            .view(batch_size, num_modalities, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)  # [batch, heads, modalities, head_dim]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_modalities, hidden_size)
        )

        # Output projection and residual connection
        output = self.out_linear(context)
        output = self.layer_norm(output + embeddings)

        # Convert back to dictionary
        updated_embeddings = {}
        for i, modality in enumerate(modality_list):
            updated_embeddings[modality] = output[:, i, :]

        return updated_embeddings


class GatedModalityFusion(nn.Module):
    """
    Gated fusion mechanism for combining modality embeddings.
    """

    def __init__(self, hidden_size: int):
        """
        Initialize gated fusion.

        Args:
            hidden_size: Hidden dimension
        """
        super().__init__()

        self.hidden_size = hidden_size

        # Gates for each modality
        self.modality_gates = nn.ModuleDict(
            {
                "mRNA": nn.Linear(hidden_size * 2, hidden_size),
                "CNV": nn.Linear(hidden_size * 2, hidden_size),
                "DNAmeth": nn.Linear(hidden_size * 2, hidden_size),
                "miRNA": nn.Linear(hidden_size * 2, hidden_size),
            }
        )

        self.fusion_layer = nn.Linear(hidden_size * 4, hidden_size)

    def forward(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
        fused_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply gated fusion.

        Args:
            modality_embeddings: Individual modality embeddings
            fused_embedding: Initial fused embedding

        Returns:
            Refined fused embedding
        """
        gated_embeddings = []

        for modality, emb in modality_embeddings.items():
            if modality in self.modality_gates:
                # Concatenate modality embedding with fused embedding
                concat = torch.cat([emb, fused_embedding], dim=-1)

                # Compute gate
                gate = torch.sigmoid(self.modality_gates[modality](concat))

                # Apply gate
                gated = gate * emb
                gated_embeddings.append(gated)

        # Concatenate all gated embeddings
        if gated_embeddings:
            all_gated = torch.cat(gated_embeddings, dim=-1)
            refined = self.fusion_layer(all_gated)

            # Residual connection
            refined = refined + fused_embedding
        else:
            refined = fused_embedding

        return refined
