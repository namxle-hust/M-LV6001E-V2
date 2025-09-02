"""
Modality-level attention fusion module.
FIXED VERSION: Properly handles batched data with multiple patients.
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
        self.modality_keys = ["mrna", "cnv", "cpg", "mirna"]

    def compute_attention_weights(
        self, modality_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention weights for each modality.
        FIXED: Now properly handles batched embeddings.

        Args:
            modality_embeddings: Dictionary of modality embeddings
                Each embedding should be [batch_size, hidden_size]

        Returns:
            Attention weights [batch_size, num_modalities]
        """
        scores = []
        batch_size = None

        for modality in self.modality_keys:
            if modality in modality_embeddings:
                emb = modality_embeddings[modality]

                # FIX: Ensure we have the right shape
                if emb.dim() == 1:
                    # If 1D, add batch dimension
                    emb = emb.unsqueeze(0)

                # Get batch size from the first modality
                if batch_size is None:
                    batch_size = emb.size(0)

                # Compute attention score for this modality
                # emb shape: [batch_size, hidden_size]
                # score shape: [batch_size, 1]
                score = self.attention_mlp(emb)
                scores.append(score)

        # Stack scores along the modality dimension
        # scores: list of [batch_size, 1] tensors
        # After cat: [batch_size, num_modalities]
        scores = torch.cat(scores, dim=1)

        # Apply temperature scaling and softmax along modality dimension
        attention_weights = F.softmax(scores / self.temperature, dim=1)

        return attention_weights

    def forward(
        self, modality_embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Fuse modality embeddings using attention.
        FIXED: Now properly handles batched data.

        Args:
            modality_embeddings: Dictionary of modality embeddings
                Each should be [batch_size, hidden_size]

        Returns:
            - Fused patient embedding [batch_size, hidden_size]
            - Attention weights [batch_size, num_modalities]
            - Dictionary of mean attention scores by modality
        """
        # Compute attention weights for each patient
        # Shape: [batch_size, num_modalities]
        attention_weights = self.compute_attention_weights(modality_embeddings)

        # Stack modality embeddings
        embeddings_list = []
        present_modalities = []
        batch_size = None

        for modality in self.modality_keys:
            if modality in modality_embeddings:
                emb = modality_embeddings[modality]

                # FIX: Ensure proper shape
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)

                if batch_size is None:
                    batch_size = emb.size(0)

                embeddings_list.append(emb)
                present_modalities.append(modality)

        # Stack embeddings: [batch_size, num_modalities, hidden_size]
        stacked_embeddings = torch.stack(embeddings_list, dim=1)

        # Apply attention weights
        # attention_weights: [batch_size, num_modalities]
        # Expand for broadcasting: [batch_size, num_modalities, 1]
        attention_weights_expanded = attention_weights.unsqueeze(-1)

        # Weighted sum: [batch_size, num_modalities, hidden_size] * [batch_size, num_modalities, 1]
        # Result: [batch_size, hidden_size]
        fused_embedding = (stacked_embeddings * attention_weights_expanded).sum(dim=1)

        # Create attention score dictionary with mean values across batch
        attention_dict = {}
        for i, modality in enumerate(present_modalities):
            # Mean attention weight for this modality across all patients in batch
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
        FIXED: Now properly handles batched embeddings.

        Args:
            modality_embeddings: Dictionary of modality embeddings
                Each should be [batch_size, hidden_size]

        Returns:
            Updated modality embeddings with cross-modality information
        """
        # Stack modality embeddings
        modality_list = list(modality_embeddings.keys())
        embeddings = []

        for m in modality_list:
            emb = modality_embeddings[m]
            # Ensure proper shape
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb)

        # Stack: [batch_size, num_modalities, hidden_size]
        embeddings = torch.stack(embeddings, dim=1)
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
                "mrna": nn.Linear(hidden_size * 2, hidden_size),
                "cnv": nn.Linear(hidden_size * 2, hidden_size),
                "cpg": nn.Linear(hidden_size * 2, hidden_size),
                "mirna": nn.Linear(hidden_size * 2, hidden_size),
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
        FIXED: Now properly handles batched embeddings.

        Args:
            modality_embeddings: Individual modality embeddings [batch_size, hidden_size]
            fused_embedding: Initial fused embedding [batch_size, hidden_size]

        Returns:
            Refined fused embedding [batch_size, hidden_size]
        """
        gated_embeddings = []

        # Ensure fused_embedding has proper shape
        if fused_embedding.dim() == 1:
            fused_embedding = fused_embedding.unsqueeze(0)

        batch_size = fused_embedding.size(0)

        for modality, emb in modality_embeddings.items():
            if modality in self.modality_gates:
                # Ensure proper shape
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)

                # Concatenate modality embedding with fused embedding
                # Both should be [batch_size, hidden_size]
                concat = torch.cat([emb, fused_embedding], dim=-1)

                # Compute gate
                gate = torch.sigmoid(self.modality_gates[modality](concat))

                # Apply gate
                gated = gate * emb
                gated_embeddings.append(gated)

        # Concatenate all gated embeddings
        if gated_embeddings:
            # all_gated: [batch_size, hidden_size * num_modalities]
            all_gated = torch.cat(gated_embeddings, dim=-1)
            refined = self.fusion_layer(all_gated)

            # Residual connection
            refined = refined + fused_embedding
        else:
            refined = fused_embedding

        return refined
