"""
Heterogeneous graph encoder using GAT or RGCN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, RGCNConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple


class HeteroGATEncoder(nn.Module):
    """Heterogeneous Graph Attention Network encoder."""

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_size: int,
        num_layers: int,
        edge_types: list,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        heads: int = 4,
    ):
        """
        Initialize HeteroGAT encoder.

        Args:
            in_channels_dict: Input channels for each node type
            hidden_size: Hidden dimension size (shared across types)
            num_layers: Number of GAT layers
            edge_types: List of edge type tuples
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            heads: Number of attention heads
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # Input projections for each node type
        self.input_projections = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.input_projections[node_type] = Linear(
                in_channels, hidden_size, bias=True
            )

        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for src, rel, dst in edge_types:
                # Use multi-head attention for intermediate layers
                if i < num_layers - 1:
                    conv_dict[(src, rel, dst)] = GATConv(
                        hidden_size,
                        hidden_size // heads,
                        heads=heads,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                else:
                    # Last layer: single head, full hidden size
                    conv_dict[(src, rel, dst)] = GATConv(
                        hidden_size,
                        hidden_size,
                        heads=1,
                        dropout=dropout,
                        add_self_loops=False,
                    )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Layer normalization modules
        if use_layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(num_layers):
                ln_dict = nn.ModuleDict()
                for node_type in in_channels_dict.keys():
                    ln_dict[node_type] = nn.LayerNorm(hidden_size)
                self.layer_norms.append(ln_dict)

        # Node type list
        self.node_types = list(in_channels_dict.keys())

    def forward(self, x_dict: Dict, edge_index_dict: Dict) -> Dict:
        """
        Forward pass through heterogeneous GAT.

        Args:
            x_dict: Node features by type
            edge_index_dict: Edge indices by relation

        Returns:
            Updated node embeddings by type
        """
        # Project input features
        h_dict = {}
        for node_type in self.node_types:
            h_dict[node_type] = self.input_projections[node_type](x_dict[node_type])
            h_dict[node_type] = F.dropout(
                h_dict[node_type], p=self.dropout, training=self.training
            )

        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            h_prev = h_dict.copy() if self.use_residual else None

            # Graph convolution
            h_dict = conv(h_dict, edge_index_dict)

            # Flatten multi-head output for intermediate layers
            if i < self.num_layers - 1:
                for node_type in self.node_types:
                    h_dict[node_type] = h_dict[node_type].view(
                        h_dict[node_type].size(0), -1
                    )

            # Layer normalization
            if self.use_layer_norm:
                for node_type in self.node_types:
                    h_dict[node_type] = self.layer_norms[i][node_type](
                        h_dict[node_type]
                    )

            # Activation and dropout
            for node_type in self.node_types:
                h_dict[node_type] = F.relu(h_dict[node_type])
                h_dict[node_type] = F.dropout(
                    h_dict[node_type], p=self.dropout, training=self.training
                )

                # Residual connection
                if self.use_residual and h_prev is not None:
                    h_dict[node_type] = h_dict[node_type] + h_prev[node_type]

        return h_dict


class HeteroRGCNEncoder(nn.Module):
    """Heterogeneous Relational GCN encoder."""

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_size: int,
        num_layers: int,
        edge_types: list,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        use_residual: bool = True,
        num_bases: Optional[int] = None,
    ):
        """
        Initialize HeteroRGCN encoder.

        Args:
            in_channels_dict: Input channels for each node type
            hidden_size: Hidden dimension size
            num_layers: Number of RGCN layers
            edge_types: List of edge type tuples
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            num_bases: Number of bases for basis decomposition
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # Input projections
        self.input_projections = nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.input_projections[node_type] = Linear(
                in_channels, hidden_size, bias=True
            )

        # RGCN layers
        self.convs = nn.ModuleList()
        num_relations = len(edge_types)

        for i in range(num_layers):
            conv_dict = {}
            for src, rel, dst in edge_types:
                conv_dict[(src, rel, dst)] = RGCNConv(
                    hidden_size,
                    hidden_size,
                    num_relations=num_relations,
                    num_bases=num_bases,
                    bias=True,
                )

            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # Layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList()
            for i in range(num_layers):
                ln_dict = nn.ModuleDict()
                for node_type in in_channels_dict.keys():
                    ln_dict[node_type] = nn.LayerNorm(hidden_size)
                self.layer_norms.append(ln_dict)

        self.node_types = list(in_channels_dict.keys())

    def forward(self, x_dict: Dict, edge_index_dict: Dict) -> Dict:
        """
        Forward pass through heterogeneous RGCN.

        Args:
            x_dict: Node features by type
            edge_index_dict: Edge indices by relation

        Returns:
            Updated node embeddings by type
        """
        # Project input features
        h_dict = {}
        for node_type in self.node_types:
            h_dict[node_type] = self.input_projections[node_type](x_dict[node_type])
            h_dict[node_type] = F.dropout(
                h_dict[node_type], p=self.dropout, training=self.training
            )

        # Apply RGCN layers
        for i, conv in enumerate(self.convs):
            h_prev = h_dict.copy() if self.use_residual else None

            # Graph convolution
            h_dict = conv(h_dict, edge_index_dict)

            # Layer normalization
            if self.use_layer_norm:
                for node_type in self.node_types:
                    h_dict[node_type] = self.layer_norms[i][node_type](
                        h_dict[node_type]
                    )

            # Activation and dropout
            for node_type in self.node_types:
                h_dict[node_type] = F.relu(h_dict[node_type])
                h_dict[node_type] = F.dropout(
                    h_dict[node_type], p=self.dropout, training=self.training
                )

                # Residual connection
                if self.use_residual and h_prev is not None:
                    h_dict[node_type] = h_dict[node_type] + h_prev[node_type]

        return h_dict


def create_encoder(config: dict, in_channels_dict: Dict[str, int], edge_types: list):
    """
    Create encoder based on configuration.

    Args:
        config: Configuration dictionary
        in_channels_dict: Input channels by node type
        edge_types: List of edge types

    Returns:
        Encoder module
    """
    encoder_config = config["model"]["encoder"]
    encoder_type = encoder_config["type"]

    common_args = {
        "in_channels_dict": in_channels_dict,
        "hidden_size": encoder_config["hidden_size"],
        "num_layers": encoder_config["num_layers"],
        "edge_types": edge_types,
        "dropout": encoder_config["dropout"],
        "use_layer_norm": encoder_config["use_layer_norm"],
        "use_residual": encoder_config["use_residual"],
    }

    if encoder_type == "HeteroGAT":
        encoder = HeteroGATEncoder(**common_args, heads=encoder_config.get("heads", 4))
    elif encoder_type == "HeteroRGCN":
        encoder = HeteroRGCNEncoder(
            **common_args, num_bases=encoder_config.get("num_bases", None)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder
