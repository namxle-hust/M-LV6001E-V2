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

# Add these imports at the top
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataio.dataset import PatientGraphDataset, custom_collate_fn
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


# def set_seed(seed: int):
#     """Set random seeds for reproducibility."""
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


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
    avg_loss = total_loss / batch_count if len(dataloader) > 0 else 0
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


def train_model(model, train_loader, val_loader, optimizer, config, device, stage="A"):
    """Train model for one fold."""
    best_val_loss = float("inf")
    best_metrics = {}

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["training"][f"stage_{stage.lower()}"]["patience"],
        verbose=False,
    )

    num_epochs = config["training"][f"stage_{stage.lower()}"]["epochs"]

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        batch_count = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data, compute_loss=True)
            loss = output["losses"]["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1

        avg_train_loss = train_loss / batch_count

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, stage)

        # Update scheduler
        scheduler.step(val_metrics["loss"])

        # Track best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {
                "best_train_loss": avg_train_loss,
                "best_val_loss": val_metrics["loss"],
                **val_metrics,
            }

    return best_metrics


def evaluate_fold(model, dataloader, device):
    """Evaluate model on validation fold."""
    model.eval()

    from src.losses.recon_edge import LinkPredictionMetrics

    metrics = {
        "recon_mse": {"mRNA": [], "CNV": [], "cpg": [], "mirna": []},
        "edge_auroc": [],
    }

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            output = model(data, compute_loss=True)

            # Get reconstruction MSE from losses
            if "recon_detail" in output["losses"]:
                for modality in ["mrna", "cnv", "cpg", "mirna"]:
                    if modality in output["losses"]["recon_detail"]:
                        value = output["losses"]["recon_detail"][modality]
                        if torch.is_tensor(value):
                            value = value.item()
                        mod_key = modality.upper() if modality != "mirna" else modality
                        metrics["recon_mse"][mod_key].append(value)

            # Calculate actual AUROC for edge prediction
            edge_types = [("cpg", "maps_to", "gene"), ("mirna", "targets", "gene")]
            for edge_type in edge_types:
                if hasattr(data[edge_type], "edge_index"):
                    src_type, _, dst_type = edge_type

                    # Get embeddings
                    z_src = output["node_embeddings"][src_type]
                    z_dst = output["node_embeddings"][dst_type]

                    # Get decoder
                    decoder = model.edge_decoder.get_decoder(edge_type)

                    # Positive edges
                    edge_index = data[edge_type].edge_index
                    if edge_index.shape[1] > 0:
                        pos_src = z_src[edge_index[0]]
                        pos_dst = z_dst[edge_index[1]]
                        pos_pred = torch.sigmoid(decoder(pos_src, pos_dst))

                        # Sample negative edges
                        neg_edge_index = model.edge_decoder.sample_negative_edges(
                            edge_index,
                            z_src.shape[0],
                            z_dst.shape[0],
                            edge_index.shape[1],
                        )

                        neg_src = z_src[neg_edge_index[0]]
                        neg_dst = z_dst[neg_edge_index[1]]
                        neg_pred = torch.sigmoid(decoder(neg_src, neg_dst))

                        # Compute AUROC
                        auroc_metrics = LinkPredictionMetrics.compute_auc_ap(
                            pos_pred, neg_pred
                        )
                        metrics["edge_auroc"].append(auroc_metrics["auroc"])

    # Average all metrics
    for modality in metrics["recon_mse"]:
        if metrics["recon_mse"][modality]:
            metrics["recon_mse"][modality] = np.mean(metrics["recon_mse"][modality])
        else:
            metrics["recon_mse"][modality] = 0.0

    metrics["edge_auroc"] = (
        np.mean(metrics["edge_auroc"]) if metrics["edge_auroc"] else 0.0
    )

    return metrics


def train_kfold(model_class, all_graphs, config, device, k_folds=5):
    """
    Perform K-fold cross-validation.

    Returns:
        Dictionary of averaged metrics across folds
    """
    # Store metrics for each fold
    fold_metrics = {
        "train_loss": [],
        "val_loss": [],
        "recon_mse": {"mRNA": [], "CNV": [], "cpg": [], "mirna": []},
        "edge_auroc": [],
    }

    # Create data module
    data_module = MultiModalDataModule(all_graphs, config, seed=config["seed"])
    data_module.setup_kfold(k_folds)

    for fold_idx in range(k_folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1}/{k_folds}")
        print(f"{'='*50}")

        # Set current fold
        data_module.set_fold(fold_idx)

        # Create new model for this fold
        model = model_class(config).to(device)

        # Get dataloaders for current fold
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        # Stage A: Pretrain
        print(f"Training Stage A...")
        model.set_training_stage("A")
        optimizer_a = optim.Adam(
            model.parameters(),
            lr=config["training"]["stage_a"]["learning_rate"],
            weight_decay=config["training"]["stage_a"]["weight_decay"],
        )
        stage_a_results = train_model(
            model, train_loader, val_loader, optimizer_a, config, device, "A"
        )

        # Stage B: Fusion
        print(f"Training Stage B...")
        model.set_training_stage("B")
        optimizer_b = optim.Adam(
            model.parameters(),
            lr=config["training"]["stage_b"]["learning_rate"],
            weight_decay=config["training"]["stage_b"]["weight_decay"],
        )
        stage_b_results = train_model(
            model, train_loader, val_loader, optimizer_b, config, device, "B"
        )

        # Store fold metrics
        fold_metrics["train_loss"].append(stage_b_results["best_train_loss"])
        fold_metrics["val_loss"].append(stage_b_results["best_val_loss"])

        # Evaluate reconstruction and edge prediction
        print(f"Evaluating fold {fold_idx + 1}...")
        eval_metrics = evaluate_fold(model, val_loader, device)
        for modality in ["mRNA", "CNV", "cpg", "mirna"]:
            fold_metrics["recon_mse"][modality].append(
                eval_metrics["recon_mse"][modality]
            )
        fold_metrics["edge_auroc"].append(eval_metrics["edge_auroc"])

        print(
            f"Fold {fold_idx + 1} - Val Loss: {stage_b_results['best_val_loss']:.4f}, "
            f"Edge AUROC: {eval_metrics['edge_auroc']:.4f}"
        )

        # Save fold checkpoint
        checkpoint_dir = os.path.join(
            config["logging"]["checkpoint_dir"], f"fold_{fold_idx}"
        )
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model, optimizer_b, 0, "B", eval_metrics, config, checkpoint_dir
        )

    # Calculate averaged metrics
    avg_metrics = {
        "val_loss": {
            "mean": np.mean(fold_metrics["val_loss"]),
            "std": np.std(fold_metrics["val_loss"]),
        },
        "recon_mse": {},
        "edge_auroc": {
            "mean": np.mean(fold_metrics["edge_auroc"]),
            "std": np.std(fold_metrics["edge_auroc"]),
        },
    }

    for modality in ["mRNA", "CNV", "cpg", "mirna"]:
        avg_metrics["recon_mse"][modality] = {
            "mean": np.mean(fold_metrics["recon_mse"][modality]),
            "std": np.std(fold_metrics["recon_mse"][modality]),
        }

    # Save K-fold results
    results_path = os.path.join(config["logging"]["log_dir"], "kfold_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "fold_metrics": fold_metrics,
                "averaged_metrics": avg_metrics,
                "n_folds": k_folds,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 50)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(
        f"Val Loss: {avg_metrics['val_loss']['mean']:.4f} ± {avg_metrics['val_loss']['std']:.4f}"
    )
    print(
        f"Edge AUROC: {avg_metrics['edge_auroc']['mean']:.4f} ± {avg_metrics['edge_auroc']['std']:.4f}"
    )
    print(f"\nPer-modality Reconstruction MSE:")
    for modality in ["mRNA", "CNV", "cpg", "mirna"]:
        mean_val = avg_metrics["recon_mse"][modality]["mean"]
        std_val = avg_metrics["recon_mse"][modality]["std"]
        print(f"  {modality}: {mean_val:.4f} ± {std_val:.4f}")

    return avg_metrics


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
    parser.add_argument(
        "--kfold",
        type=int,
        default=5,
        help="Number of folds for cross-validation (0=no k-fold)",
    )

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

    print(f"Args: {args}")

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

    # Check if K-fold training is requested
    if args.kfold > 0:
        print(f"\n=== K-FOLD CROSS-VALIDATION (K={args.kfold}) ===")

        # Run K-fold cross-validation
        kfold_results = train_kfold(
            MultiModalGNNWithDecoders, all_graphs, config, device, k_folds=args.kfold
        )

        print("\n=== TRAINING FINAL MODEL ON ALL PATIENTS ===")

        # Create fresh model for final training on all data
        model = MultiModalGNNWithDecoders(config).to(device)

        # Use all graphs for training (with small validation set)
        n_val = min(20, len(all_graphs) // 10)  # Small validation set
        train_graphs = all_graphs[:-n_val] if n_val > 0 else all_graphs
        val_graphs = all_graphs[-n_val:] if n_val > 0 else all_graphs[-1:]

        # Create datasets
        train_dataset = PatientGraphDataset(train_graphs)
        val_dataset = PatientGraphDataset(val_graphs)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
            collate_fn=custom_collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=config["training"]["pin_memory"],
            collate_fn=custom_collate_fn,
        )

        test_loader = None  # No test set when using all data

    else:
        # Standard training with train/val/test split
        print("\n=== STANDARD TRAINING WITH TRAIN/VAL/TEST SPLIT ===")

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

    # Final evaluation
    if test_loader is not None:
        print("\n=== Final Evaluation on Test Set ===")
        test_metrics = validate_epoch(model, test_loader, device, "B")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
    else:
        print("\n=== Final Evaluation on Validation Set ===")
        final_metrics = validate_epoch(model, val_loader, device, "B")
        print(f"Final Loss: {final_metrics['loss']:.4f}")

    # Export embeddings for ALL patients
    print("\nExporting embeddings for all patients...")
    all_loader = DataLoader(
        PatientGraphDataset(all_graphs),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        collate_fn=custom_collate_fn,
    )
    export_embeddings(model, all_loader, device, config["logging"]["tensors_dir"])

    # Save final metrics
    metrics_file = os.path.join(config["logging"]["log_dir"], "final_metrics.json")
    final_metrics_dict = {
        "num_parameters": num_params,
        "config": config,
    }

    if test_loader is not None:
        final_metrics_dict["test_metrics"] = test_metrics
    else:
        final_metrics_dict["final_metrics"] = final_metrics

    if args.kfold > 0:
        final_metrics_dict["kfold_results"] = kfold_results

    with open(metrics_file, "w") as f:
        json.dump(final_metrics_dict, f, indent=2)

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
