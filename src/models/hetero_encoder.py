# src/models/hetero_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional


class HeteroGATEncoder(nn.Module):
    """
    Heterogeneous Graph Attention Network encoder with relation-specific parameters.
    """
    
    def __init__(self, 
                 in_channels_dict: Dict[str, int],
                 hidden_channels: int,
                 num_layers: int,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 layer_norm: bool = True,
                 concat_heads: bool = False):
        super().__init__()
        
        self.in_channels_dict = in_channels_dict
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.concat_heads = concat_heads
        
        # Input projections for each node type
        self.input_projections = nn.ModuleDict({
            node_type: Linear(in_channels, hidden_channels)
            for node_type, in_channels in in_channels_dict.items()
        })
        
        # Build heterogeneous GNN layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            
            # Define convolutions for each edge type
            # CpG -> Gene
            conv_dict[('cpg', 'maps_to', 'gene')] = GATConv(
                hidden_channels,
                hidden_channels // num_heads if not concat_heads else hidden_channels,
                heads=num_heads,
                concat=concat_heads,
                dropout=dropout,
                add_self_loops=False
            )
            
            # miRNA -> Gene
            conv_dict[('mirna', 'targets', 'gene')] = GATConv(
                hidden_channels,
                hidden_channels // num_heads if not concat_heads else hidden_channels,
                heads=num_heads,
                concat=concat_heads,
                dropout=dropout,
                add_self_loops=False
            )
            
            # Gene -> Gene (optional PPI)
            conv_dict[('gene', 'ppi', 'gene')] = GATConv(
                hidden_channels,
                hidden_channels // num_heads if not concat_heads else hidden_channels,
                heads=num_heads,
                concat=concat_heads,
                dropout=dropout,
                add_self_loops=True
            )
            
            # Add reverse edges for message passing
            conv_dict[('gene', 'rev_maps_to', 'cpg')] = GATConv(
                hidden_channels,
                hidden_channels // num_heads if not concat_heads else hidden_channels,
                heads=num_heads,
                concat=concat_heads,
                dropout=dropout,
                add_self_loops=False
            )
            
            conv_dict[('gene', 'rev_targets', 'mirna')] = GATConv(
                hidden_channels,
                hidden_channels // num_heads if not concat_heads else hidden_channels,
                heads=num_heads,
                concat=concat_heads,
                dropout=dropout,
                add_self_loops=False
            )
            
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)
        
        # Layer normalization for each node type
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.ModuleDict({
                    node_type: nn.LayerNorm(hidden_channels)
                    for node_type in in_channels_dict.keys()
                })
                for _ in range(num_layers)
            ])
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through heterogeneous GNN.
        
        Args:
            x_dict: Node features for each node type
            edge_index_dict: Edge indices for each edge type
            edge_weight_dict: Optional edge weights
            
        Returns:
            Updated node embeddings for each node type
        """
        
        # Initial projection
        h_dict = {
            node_type: self.input_projections[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Apply dropout to input
        h_dict = {
            node_type: self.dropout_layer(h)
            for node_type, h in h_dict.items()
        }
        
        # Message passing layers
        for i, conv in enumerate(self.convs):
            h_dict_new = conv(h_dict, edge_index_dict)
            
            # Apply layer norm if enabled
            if self.layer_norm:
                h_dict_new = {
                    node_type: self.layer_norms[i][node_type](h)
                    for node_type, h in h_dict_new.items()
                }
            
            # Apply activation and dropout
            h_dict_new = {
                node_type: F.relu(h)
                for node_type, h in h_dict_new.items()
            }
            
            h_dict_new = {
                node_type: self.dropout_layer(h)
                for node_type, h in h_dict_new.items()
            }
            
            # Residual connection (if not first layer)
            if i > 0:
                h_dict = {
                    node_type: h_dict[node_type] + h_dict_new[node_type]
                    for node_type in h_dict.keys()
                }
            else:
                h_dict = h_dict_new
        
        return h_dict


# src/models/modality_pool.py
class ModalityPooling(nn.Module):
    """
    Pool node embeddings to get modality-level representations.
    Handles the split between mRNA and CNV from gene nodes.
    """
    
    def __init__(self, 
                 hidden_channels: int,
                 pooling_type: str = 'mean',
                 separate_gene_channels: bool = False):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.pooling_type = pooling_type
        self.separate_gene_channels = separate_gene_channels
        
        if separate_gene_channels:
            # If mRNA and CNV are stored separately
            self.mrna_projection = nn.Linear(hidden_channels, hidden_channels)
            self.cnv_projection = nn.Linear(hidden_channels, hidden_channels)
        else:
            # If they're concatenated, split them
            self.gene_split = nn.ModuleDict({
                'mrna': nn.Linear(hidden_channels, hidden_channels),
                'cnv': nn.Linear(hidden_channels, hidden_channels)
            })
        
        # Attention pooling layers if needed
        if pooling_type == 'attention':
            self.attention_layers = nn.ModuleDict({
                'gene': nn.Linear(hidden_channels, 1),
                'cpg': nn.Linear(hidden_channels, 1),
                'mirna': nn.Linear(hidden_channels, 1)
            })
    
    def forward(self, 
                h_dict: Dict[str, torch.Tensor],
                batch_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Pool node embeddings to modality vectors.
        
        Returns:
            Dictionary with keys: 'mRNA', 'CNV', 'DNAmeth', 'miRNA'
        """
        modality_vectors = {}
        
        # Process gene nodes -> mRNA and CNV
        gene_embeddings = h_dict['gene']
        
        if self.separate_gene_channels:
            # Already separate
            mrna_embeddings = self.mrna_projection(gene_embeddings)
            cnv_embeddings = self.cnv_projection(gene_embeddings)
        else:
            # Split using learned projections
            mrna_embeddings = self.gene_split['mrna'](gene_embeddings)
            cnv_embeddings = self.gene_split['cnv'](gene_embeddings)
        
        # Pool mRNA
        modality_vectors['mRNA'] = self._pool(mrna_embeddings, batch_dict.get('gene') if batch_dict else None, 'gene')
        
        # Pool CNV
        modality_vectors['CNV'] = self._pool(cnv_embeddings, batch_dict.get('gene') if batch_dict else None, 'gene')
        
        # Pool DNAmeth from CpG nodes
        modality_vectors['DNAmeth'] = self._pool(h_dict['cpg'], batch_dict.get('cpg') if batch_dict else None, 'cpg')
        
        # Pool miRNA
        modality_vectors['miRNA'] = self._pool(h_dict['mirna'], batch_dict.get('mirna') if batch_dict else None, 'mirna')
        
        return modality_vectors
    
    def _pool(self, embeddings: torch.Tensor, batch: Optional[torch.Tensor], node_type: str) -> torch.Tensor:
        """Apply pooling operation."""
        if self.pooling_type == 'mean':
            if batch is not None:
                # Batched pooling
                from torch_geometric.nn import global_mean_pool
                return global_mean_pool(embeddings, batch)
            else:
                # Single graph
                return embeddings.mean(dim=0, keepdim=True)
                
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attention_scores = self.attention_layers[node_type](embeddings)
            attention_weights = F.softmax(attention_scores, dim=0)
            
            if batch is not None:
                # Batched attention pooling
                from torch_geometric.nn import global_add_pool
                weighted_embeddings = embeddings * attention_weights
                return global_add_pool(weighted_embeddings, batch)
            else:
                # Single graph
                return (embeddings * attention_weights).sum(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")