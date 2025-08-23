# src/__init__.py
"""Multi-Omics GNN Package"""

__version__ = "1.0.0"


# src/dataio/__init__.py
from .load_features import FeatureLoader
from .load_edges import EdgeLoader
from .build_patient_graph import PatientGraphBuilder
from .dataset import MultiOmicsDataset, get_dataloader
from .collate import custom_collate

__all__ = [
    "FeatureLoader",
    "EdgeLoader",
    "PatientGraphBuilder",
    "MultiOmicsDataset",
    "get_dataloader",
    "custom_collate",
]


# src/models/__init__.py
from .hetero_encoder import HeteroGATEncoder
from .modality_pool import ModalityPooling
from .modality_attention import ModalityAttention
from .decoders import FeatureDecoder, EdgeDecoder
from .multiomics_gnn import MultiOmicsGNN

__all__ = [
    "HeteroGATEncoder",
    "ModalityPooling",
    "ModalityAttention",
    "FeatureDecoder",
    "EdgeDecoder",
    "MultiOmicsGNN",
]


# src/losses/__init__.py
from .recon_feature import FeatureReconstructionLoss
from .recon_edge import EdgeReconstructionLoss
from .consistency import ConsistencyLoss
from .entropy_reg import AttentionEntropyLoss
from .weighting import compute_modality_weights

__all__ = [
    "FeatureReconstructionLoss",
    "EdgeReconstructionLoss",
    "ConsistencyLoss",
    "AttentionEntropyLoss",
    "compute_modality_weights",
]


# src/train/__init__.py
from .trainer import Trainer
from .metrics import compute_metrics, MetricTracker
from .scheduler import get_scheduler

__all__ = ["Trainer", "compute_metrics", "MetricTracker", "get_scheduler"]
