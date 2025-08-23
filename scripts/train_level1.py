# scripts/train_level1.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Import all modules
from src.dataio.load_features import FeatureLoader
from src.dataio.load_edges import EdgeLoader
from src.dataio.build_patient_graph import PatientGraphBuilder
from src.dataio.dataset import MultiOmicsDataset, get_dataloader
from src.dataio.collate import custom_collate

from src.models.hetero_encoder import HeteroGATEncoder
from src.models.modality_pool import ModalityPooling
from src.models.modality_attention import ModalityAttention
from src.models.decoders import FeatureDecoder, EdgeDecoder
from src.models.multiomics_gnn import MultiOmicsGNN

from src.losses.recon_feature import FeatureReconstructionLoss
from src.losses.recon_edge import EdgeReconstructionLoss
from src.losses.consistency import ConsistencyLoss
from src.losses.entropy_reg import AttentionEntropyLoss
from src.losses.weighting import compute_modality_weights

from src.train.trainer import Trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Recursively merge configuration dictionaries."""
    for key, value in override_config.items():
        if (
            key in base_config
            and isinstance(base_config[key], dict)
            and isinstance(value, dict)
        ):
            base_config[key] = merge_configs(base_config[key], value)
        else:
            base_config[key] = value
    return base_config


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Multi-Omics GNN Level-1")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode with reduced data"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config("config/default.yaml")  # Load defaults first

    if args.config != "config/train.yaml":
        train_config = load_config(args.config)
        config = merge_configs(config, train_config)

    # Override with command line arguments
    if args.device:
        config["device"] = args.device
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.seed:
        config["seed"] = args.seed

    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(config["seed"])
    print(f"Random seed: {config['seed']}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = MultiOmicsDataset(config, split="train")
    val_dataset = MultiOmicsDataset(config, split="val")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Get data statistics
    num_features_dict = train_dataset.get_num_features()
    node_counts = train_dataset.get_node_counts()

    print(f"\nNode counts:")
    for node_type, count in node_counts.items():
        print(f"  {node_type}: {count} nodes, {num_features_dict[node_type]} features")

    # Compute modality weights
    modality_weights = compute_modality_weights(node_counts)
    config["losses"]["modality_weights"] = modality_weights

    print(f"\nModality weights:")
    for modality, weight in modality_weights.items():
        print(f"  {modality}: {weight:.3f}")

    # Create data loaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging, increase for production
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\nInitializing model...")
    model = MultiOmicsGNN(config, num_features_dict)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, config, device)

    # Train model
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)

    print("\nTraining completed successfully!")
    print(f"Best model saved to: {config['outputs']['checkpoint_path']}")
    print(f"Patient embeddings saved to: {config['outputs']['embeddings_path']}")
    print(f"Attention weights saved to: {config['outputs']['attention_weights_path']}")


if __name__ == "__main__":
    main()
