"""
Main training script for Level-1 multi-modal heterogeneous GNN.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataio.load_features import FeatureLoader, validate_features
from src.dataio.load_edges import EdgeLoader, validate_edges
from src.dataio.build_patient_graph import PatientGraphBuilder
from src.dataio.dataset import MultiModalDataModule
from src.models.multimodal_gnn import MultiModalGNNWithDecoders
from src.utils.seed import set_seed
from src.utils.logging import setup_logging, log_metrics


def load_config(config_path: str, overrides: dict = None) -> dict:
    """Load configuration from YAML file with optional overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply command-line overrides
    if overrides:
        for key, value in overrides.items():
            # Navigate nested dictionary
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    stage: str,
    writer: SummaryWriter = None,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    loss_components = {}
    batch_count = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - Stage {stage}")

    for batch_idx, data in enumerate(progress_bar):
        data = data.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data, compute_loss=True)

        loss = output["losses"]["total"]

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track losses
        total_loss += loss.item()
        batch_count += 1

        # Accumulate loss components
        for key, value in output["losses"].items():
            if key != "total" and not key.endswith("_detail"):
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += (
                    value.item() if torch.is_tensor(value) else value
                )

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar(f"Loss/train_{stage}", loss.item(), global_step)

    # Average losses
    avg_loss = total_loss / batch_count
    avg_components = {k: v / batch_count for k, v in loss_components.items()}

    return {"loss": avg_loss, **avg_components}


def validate_epoch(
    model: nn.Module, dataloader, device: torch.device, stage: str
) -> dict:
    """Validate for one epoch."""
    model.eval()

    total_loss = 0
    loss_components = {}
    batch_count = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Validation - Stage {stage}"):
            data = data.to(device)

            # Forward pass
            output = model(data, compute_loss=True)

            loss = output["losses"]["total"]

            # Track losses
            total_loss += loss.item()
            batch_count += 1

            # Accumulate loss components
            for key, value in output["losses"].items():
                if key != "total" and not key.endswith("_detail"):
                    if key not in loss_components:
                        loss_components[key] = 0
                    loss_components[key] += (
                        value.item() if torch.is_tensor(value) else value
                    )

    # Average losses
    avg_loss = total_loss / batch_count
    avg_components = {k: v / batch_count for k, v in loss_components.items()}

    return {"loss": avg_loss, **avg_components}


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    stage: str,
    metrics: dict,
    config: dict,
    checkpoint_dir: str,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save with descriptive filename
    filename = f"checkpoint_stage{stage}_epoch{epoch:03d}.pt"
    filepath = os.path.join(checkpoint_dir, filename)

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

    return filepath


def export_embeddings(
    model: nn.Module, dataloader, device: torch.device, output_dir: str
):
    """Export patient and modality embeddings."""
    model.eval()

    all_patient_embeddings = []
    all_modality_embeddings = {"mRNA": [], "CNV": [], "DNAmeth": [], "miRNA": []}
    all_attention_weights = []
    patient_ids = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Exporting embeddings"):
            data = data.to(device)

            output = model(data, compute_loss=False)

            # Patient embeddings
            all_patient_embeddings.append(output["fused_embedding"].cpu())

            # Modality embeddings
            for modality in all_modality_embeddings.keys():
                all_modality_embeddings[modality].append(
                    output["modality_embeddings"][modality].cpu()
                )

            # Attention weights
            if output["attention_weights"] is not None:
                all_attention_weights.append(output["attention_weights"].cpu())

            # Patient IDs
            patient_ids.extend(data.patient_ids)

    # Concatenate
    patient_embeddings = torch.cat(all_patient_embeddings, dim=0)

    modality_embeddings = {}
    for modality in all_modality_embeddings.keys():
        modality_embeddings[modality] = torch.cat(
            all_modality_embeddings[modality], dim=0
        )

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save embeddings
    torch.save(patient_embeddings, os.path.join(output_dir, "patient_embeddings.pt"))
    torch.save(modality_embeddings, os.path.join(output_dir, "modality_embeddings.pt"))

    # Save attention weights as CSV
    if all_attention_weights:
        attention_weights = torch.cat(all_attention_weights, dim=0).numpy()
        attention_df = pd.DataFrame(
            attention_weights,
            index=patient_ids,
            columns=["mRNA", "CNV", "DNAmeth", "miRNA"],
        )
        attention_df.to_csv(os.path.join(output_dir, "attention_weights.csv"))

    print(f"Embeddings exported to {output_dir}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train Level-1 Multi-Modal Heterogeneous GNN"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--seed", type=int, help="Override random seed")

    args = parser.parse_args()

    # Build overrides dictionary
    overrides = {}
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size
    if args.lr:
        overrides["training.stage_a.learning_rate"] = args.lr
    if args.epochs:
        overrides["training.stage_a.epochs"] = args.epochs
    if args.device:
        overrides["device"] = args.device
    if args.seed:
        overrides["seed"] = args.seed

    # Load configuration
    config = load_config(args.config, overrides)

    # Set seed
    set_seed(config["seed"])

    # Setup device
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    print(f"Using device: {device}")

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config["logging"]["log_dir"], f"run_{timestamp}")
    writer = SummaryWriter(log_dir)

    print("Loading data...")

    # Load features
    feature_loader = FeatureLoader(config["data"]["features_dir"])
    features_dict = feature_loader.load_all_features(config)
    validate_features(features_dict)

    # Load edges
    edge_loader = EdgeLoader(
        config["data"]["edges_dir"], config["data"]["default_edge_weight"]
    )
    edges_dict = edge_loader.load_all_edges(config, features_dict["node_ids"])
    validate_edges(edges_dict, features_dict["num_nodes"])

    # Build patient graphs
    print("Building patient graphs...")
    graph_builder = PatientGraphBuilder(features_dict, edges_dict)
    all_graphs = graph_builder.build_all_graphs()

    # Create data module
    data_module = MultiModalDataModule(all_graphs, config, seed=config["seed"])

    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Create model
    print(f"Creating model for device: {device}...")
    model = MultiModalGNNWithDecoders(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["stage_a"]["learning_rate"],
        weight_decay=config["training"]["stage_a"]["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["training"]["stage_a"]["patience"],
        verbose=True,
    )

    # Training Stage A: Pretrain with reconstruction
    print("\n=== Stage A: Pretraining with Reconstruction ===")
    model.set_training_stage("A")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config["training"]["stage_a"]["epochs"] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, "A", writer
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, "A")

        # Log metrics
        print(
            f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )

        writer.add_scalar("Loss/train_A", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val_A", val_metrics["loss"], epoch)

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best checkpoint
            save_checkpoint(
                model,
                optimizer,
                epoch,
                "A",
                val_metrics,
                config,
                config["logging"]["checkpoint_dir"],
            )
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Training Stage B: Fusion with attention
    print("\n=== Stage B: Fusion with Attention ===")
    model.set_training_stage("B")

    # Reset optimizer for stage B
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["stage_b"]["learning_rate"],
        weight_decay=config["training"]["stage_b"]["weight_decay"],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["training"]["stage_b"]["patience"],
        verbose=True,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config["training"]["stage_b"]["epochs"] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, "B", writer
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, "B")

        # Log metrics
        print(
            f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )

        writer.add_scalar("Loss/train_B", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val_B", val_metrics["loss"], epoch)

        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best checkpoint
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                epoch,
                "B",
                val_metrics,
                config,
                config["logging"]["checkpoint_dir"],
            )

            # Also save as best model
            best_path = os.path.join(
                config["logging"]["checkpoint_dir"], "level1_best.pt"
            )
            torch.save(torch.load(checkpoint_path), best_path)

        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Final evaluation on test set
    print("\n=== Final Evaluation ===")
    test_metrics = validate_epoch(model, test_loader, device, "B")
    print(f"Test Loss: {test_metrics['loss']:.4f}")

    # Export embeddings
    print("\nExporting embeddings...")
    export_embeddings(model, test_loader, device, config["logging"]["tensors_dir"])

    # Save final metrics
    metrics_file = os.path.join(config["logging"]["log_dir"], "final_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(
            {
                "test_metrics": test_metrics,
                "num_parameters": num_params,
                "config": config,
            },
            f,
            indent=2,
        )

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
