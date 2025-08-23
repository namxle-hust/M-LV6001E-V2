# src/train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json


class Trainer:
    """Trainer class for Multi-Omics GNN."""

    def __init__(self, model: nn.Module, config: dict, device: torch.device):

        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config["training"]["factor"],
            patience=config["training"]["patience"],
        )

        # Logging
        self.writer = SummaryWriter(config["logging"]["log_dir"])
        self.checkpoint_dir = Path(config["checkpoint"]["save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.stage = "pretrain"

        # Metrics storage
        self.attention_weights_history = []

    def train_epoch(self, train_loader, stage: str = "full") -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {}
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training {stage}")

        for batch in progress_bar:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            results = self.model(batch, return_attention=(stage == "full"))

            # Compute loss
            loss, loss_dict = self.model.compute_loss(results, batch, stage)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value.item()

            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

            # Store attention weights for analysis
            if stage == "full" and "attention_weights" in results:
                self._store_attention_weights(
                    results["attention_weights"], batch.patient_ids
                )

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        val_losses = {}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(self.device)

                # Forward pass
                results = self.model(batch, return_attention=True)

                # Compute loss
                loss, loss_dict = self.model.compute_loss(
                    results, batch, stage="full" if self.stage == "full" else "pretrain"
                )

                # Accumulate losses
                for key, value in loss_dict.items():
                    if key not in val_losses:
                        val_losses[key] = 0
                    val_losses[key] += value.item()

                num_batches += 1

        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def train(self, train_loader, val_loader, num_epochs: Optional[int] = None):
        """Main training loop with two-stage training."""

        if num_epochs is None:
            num_epochs = self.config["training"]["num_epochs"]

        pretrain_epochs = self.config["training"]["pretrain_epochs"]

        # Stage A: Pretrain with reconstruction losses only
        print("\n=== Stage A: Pretraining ===")
        self.stage = "pretrain"

        for epoch in range(pretrain_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch(train_loader, stage="pretrain")

            # Validate
            val_losses = self.validate(val_loader)

            # Log metrics
            self._log_metrics(train_losses, val_losses, epoch)

            # Print progress
            print(f"Epoch {epoch+1}/{pretrain_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")

            # Check for plateau
            self.scheduler.step(val_losses["total"])

            if self.optimizer.param_groups[0]["lr"] < 1e-6:
                print("Learning rate too small, ending pretraining")
                break

        # Stage B: Full training with all losses
        print("\n=== Stage B: Full Training ===")
        self.stage = "full"
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        for epoch in range(pretrain_epochs, num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch(train_loader, stage="full")

            # Validate
            val_losses = self.validate(val_loader)

            # Log metrics
            self._log_metrics(train_losses, val_losses, epoch)

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_losses["total"])

            # Early stopping check
            if (
                val_losses["total"]
                < self.best_val_loss - self.config["training"]["min_delta"]
            ):
                self.best_val_loss = val_losses["total"]
                self.patience_counter = 0

                # Save best model
                self._save_checkpoint(epoch, val_losses, is_best=True)
            else:
                self.patience_counter += 1

                if (
                    self.patience_counter
                    >= self.config["training"]["early_stopping_patience"]
                ):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Regular checkpoint
            if (epoch + 1) % self.config["checkpoint"]["save_frequency"] == 0:
                self._save_checkpoint(epoch, val_losses, is_best=False)

        # Save final outputs
        self._save_final_outputs(train_loader)

        print("\nTraining completed!")

    def _log_metrics(self, train_losses: Dict, val_losses: Dict, epoch: int):
        """Log metrics to TensorBoard."""
        # Log losses
        for key, value in train_losses.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)

        for key, value in val_losses.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)

        # Log learning rate
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

        # Log gradient norms
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.writer.add_scalar("gradient_norm", total_norm, epoch)

    def _store_attention_weights(self, attention_weights: Dict, patient_ids: list):
        """Store attention weights for later analysis."""
        for i, patient_id in enumerate(patient_ids):
            weights = {
                "patient_id": patient_id,
                "epoch": self.current_epoch,
                "weights": {
                    modality: attention_weights[modality][i].cpu().numpy().tolist()
                    for modality in attention_weights
                },
            }
            self.attention_weights_history.append(weights)

    def _save_checkpoint(self, epoch: int, val_losses: Dict, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_losses": val_losses,
            "config": self.config,
        }

        if is_best:
            path = self.checkpoint_dir / "level1_best.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def _save_final_outputs(self, data_loader):
        """Save final embeddings and attention weights."""
        self.model.eval()

        all_patient_embeddings = []
        all_modality_embeddings = {m: [] for m in ["mRNA", "CNV", "DNAmeth", "miRNA"]}
        all_attention_weights = []
        all_patient_ids = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating final outputs"):
                batch = batch.to(self.device)

                results = self.model(batch, return_attention=True)

                # Store embeddings
                all_patient_embeddings.append(results["patient_embedding"].cpu())

                for modality in all_modality_embeddings:
                    all_modality_embeddings[modality].append(
                        results["modality_vectors"][modality].cpu()
                    )

                # Store attention weights
                for i, patient_id in enumerate(batch.patient_ids):
                    weights = {
                        "patient_id": patient_id,
                        "mRNA": results["attention_weights"]["mRNA"][i].cpu().numpy(),
                        "CNV": results["attention_weights"]["CNV"][i].cpu().numpy(),
                        "DNAmeth": results["attention_weights"]["DNAmeth"][i]
                        .cpu()
                        .numpy(),
                        "miRNA": results["attention_weights"]["miRNA"][i].cpu().numpy(),
                    }
                    all_attention_weights.append(weights)
                    all_patient_ids.append(patient_id)

        # Concatenate and save
        outputs_dir = Path(self.config["outputs"]["embeddings_path"]).parent
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Patient embeddings
        patient_embeddings = torch.cat(all_patient_embeddings, dim=0)
        torch.save(
            {"embeddings": patient_embeddings, "patient_ids": all_patient_ids},
            self.config["outputs"]["embeddings_path"],
        )

        # Modality embeddings
        modality_embeddings = {
            m: torch.cat(embs, dim=0) for m, embs in all_modality_embeddings.items()
        }
        torch.save(
            {"embeddings": modality_embeddings, "patient_ids": all_patient_ids},
            self.config["outputs"]["modality_embeddings_path"],
        )

        # Attention weights as CSV
        import pandas as pd

        attention_df = pd.DataFrame(all_attention_weights)
        attention_df.to_csv(
            self.config["outputs"]["attention_weights_path"], index=False
        )

        print(f"Saved final outputs to {outputs_dir}")
