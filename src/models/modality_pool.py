"""
Per-modality pooling to obtain modality-specific embeddings.
FIXED VERSION: Properly handles batched data with multiple patients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import AttentionalAggregation
from typing import Dict, Optional, Tuple


class ModalityPooling(nn.Module):
    """Pool node embeddings to get modality-level representations."""

    def __init__(
        self,
        hidden_size: int,
        pool_type: str = "mean",
        use_projection_heads: bool = True,
    ):
        """
        Initialize modality pooling module.

        Args:
            hidden_size: Hidden dimension size
            pool_type: Type of pooling ('mean', 'sum', 'max', 'attention')
            use_projection_heads: Whether to use separate projection heads for mrna/cnv
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.pool_type = pool_type
        self.use_projection_heads = use_projection_heads

        # Projection heads for splitting gene embeddings into mrna and cnv
        if use_projection_heads:
            self.mrna_projection = nn.Linear(hidden_size, hidden_size)
            self.cnv_projection = nn.Linear(hidden_size, hidden_size)

        # Attention pooling if selected
        if pool_type == "attention":
            self.gene_attention = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1),
                )
            )
            self.cpg_attention = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1),
                )
            )
            self.mirna_attention = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1),
                )
            )

    def pool_nodes(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        pool_fn: str = "mean",
    ) -> torch.Tensor:
        """
        Pool node features.

        Args:
            x: Node features [n_nodes, hidden_size]
            batch: Batch assignment for nodes
            pool_fn: Pooling function name

        Returns:
            Pooled features [batch_size, hidden_size]
        """
        # FIX: Properly handle batch assignments
        # If batch is None, we need to handle single vs batched graphs correctly
        if batch is None:
            # If x has no nodes, return zeros
            if x.size(0) == 0:
                return torch.zeros(1, x.size(1), device=x.device)
            # For truly single graph case, create batch assignment
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if pool_fn == "mean":
            return global_mean_pool(x, batch)
        elif pool_fn == "sum":
            return global_add_pool(x, batch)
        elif pool_fn == "max":
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling function: {pool_fn}")

    def forward(
        self,
        node_embeddings: Dict[str, torch.Tensor],
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Pool node embeddings to get modality embeddings.

        FIXED: Now properly pools per-patient when batch_dict is provided.

        Args:
            node_embeddings: Node embeddings by type
            batch_dict: Batch assignments by node type (for batched graphs)

        Returns:
            Dictionary of modality embeddings:
                - 'mrna': mrna expression embedding [batch_size, hidden_size]
                - 'cnv': Copy number variation embedding [batch_size, hidden_size]
                - 'dnameth': DNA methylation embedding [batch_size, hidden_size]
                - 'mirna': mirna expression embedding [batch_size, hidden_size]
        """
        modality_embeddings = {}

        # FIX: Determine batch size from batch assignments
        if batch_dict is not None and "gene" in batch_dict:
            # Get the actual batch size (number of unique patients)
            batch_size = (
                batch_dict["gene"].max().item() + 1
                if batch_dict["gene"].numel() > 0
                else 1
            )
        else:
            batch_size = 1

        # Handle gene nodes (split into mrna and cnv)
        gene_emb = node_embeddings["gene"]
        gene_batch = batch_dict["gene"] if batch_dict else None

        if self.use_projection_heads:
            # Apply separate projection heads before pooling
            mrna_emb = self.mrna_projection(gene_emb)
            cnv_emb = self.cnv_projection(gene_emb)

            # Pool separately - this will now pool per-patient
            if self.pool_type == "attention":
                modality_embeddings["mrna"] = self.gene_attention(mrna_emb, gene_batch)
                modality_embeddings["cnv"] = self.gene_attention(cnv_emb, gene_batch)
            else:
                modality_embeddings["mrna"] = self.pool_nodes(
                    mrna_emb, gene_batch, self.pool_type
                )
                modality_embeddings["cnv"] = self.pool_nodes(
                    cnv_emb, gene_batch, self.pool_type
                )
        else:
            # Pool gene embeddings directly (combined mrna+cnv)
            if self.pool_type == "attention":
                pooled_gene = self.gene_attention(gene_emb, gene_batch)
            else:
                pooled_gene = self.pool_nodes(gene_emb, gene_batch, self.pool_type)

            # Split after pooling (simpler but less flexible)
            modality_embeddings["mrna"] = pooled_gene
            modality_embeddings["cnv"] = pooled_gene

        # Handle cpg nodes (DNA methylation)
        cpg_emb = node_embeddings["cpg"]
        cpg_batch = batch_dict["cpg"] if batch_dict else None

        if self.pool_type == "attention":
            modality_embeddings["dnameth"] = self.cpg_attention(cpg_emb, cpg_batch)
        else:
            modality_embeddings["dnameth"] = self.pool_nodes(
                cpg_emb, cpg_batch, self.pool_type
            )

        # Handle mirna nodes
        mirna_emb = node_embeddings["mirna"]
        mirna_batch = batch_dict["mirna"] if batch_dict else None

        if self.pool_type == "attention":
            modality_embeddings["mirna"] = self.mirna_attention(mirna_emb, mirna_batch)
        else:
            modality_embeddings["mirna"] = self.pool_nodes(
                mirna_emb, mirna_batch, self.pool_type
            )

        # FIX: Ensure all modality embeddings have the correct batch dimension
        # Each should be [batch_size, hidden_size]
        for modality in modality_embeddings:
            if modality_embeddings[modality].dim() == 1:
                # If we got a 1D tensor, reshape to [1, hidden_size]
                modality_embeddings[modality] = modality_embeddings[modality].unsqueeze(
                    0
                )

            # Verify shape
            expected_batch_size = batch_size
            actual_batch_size = modality_embeddings[modality].size(0)
            if actual_batch_size != expected_batch_size:
                print(
                    f"Warning: {modality} has batch size {actual_batch_size}, expected {expected_batch_size}"
                )

        return modality_embeddings


class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling with node-type specific pooling first,
    then modality-level aggregation.
    """

    def __init__(
        self,
        hidden_size: int,
        node_pool_type: str = "mean",
        modality_pool_type: str = "attention",
    ):
        """
        Initialize hierarchical pooling.

        Args:
            hidden_size: Hidden dimension
            node_pool_type: Pooling for nodes within type
            modality_pool_type: Pooling across node types for modality
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.node_pool_type = node_pool_type
        self.modality_pool_type = modality_pool_type

        # Node-level pooling
        self.node_pooling = ModalityPooling(
            hidden_size=hidden_size, pool_type=node_pool_type, use_projection_heads=True
        )

        # Optional: Additional transformation after pooling
        self.modality_transforms = nn.ModuleDict(
            {
                "mrna": nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ),
                "cnv": nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ),
                "dnameth": nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ),
                "mirna": nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ),
            }
        )

    def forward(
        self,
        node_embeddings: Dict[str, torch.Tensor],
        batch_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for hierarchical pooling.

        Args:
            node_embeddings: Node embeddings by type
            batch_dict: Batch assignments

        Returns:
            - Raw modality embeddings
            - Transformed modality embeddings
        """
        # Get raw pooled embeddings - now properly per-patient
        raw_modality_emb = self.node_pooling(node_embeddings, batch_dict)

        # Apply modality-specific transformations
        transformed_emb = {}
        for modality, emb in raw_modality_emb.items():
            transformed_emb[modality] = self.modality_transforms[modality](emb)

        return raw_modality_emb, transformed_emb
