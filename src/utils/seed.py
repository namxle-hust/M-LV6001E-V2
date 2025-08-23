# src/utils/seed.py
import torch
import numpy as np
import random
import os


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
        deterministic: Whether to enforce deterministic algorithms
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# src/utils/logging.py
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import torch
from typing import Dict, Any, Optional


class MetricLogger:
    """Logger for training metrics and model information."""

    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment name with timestamp
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"

        # Setup logger
        self._setup_logger()

        # Metrics storage
        self.metrics_history = []

    def _setup_logger(self):
        """Setup Python logger."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_config(self, config: dict):
        """Log configuration."""
        self.logger.info("Configuration:")
        self.logger.info(json.dumps(config, indent=2))

        # Save config to file
        config_file = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str = "train"):
        """Log metrics for an epoch."""
        metric_str = f"Epoch {epoch} [{phase}]: "
        metric_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(metric_str)

        # Store metrics
        self.metrics_history.append({"epoch": epoch, "phase": phase, **metrics})

    def log_model_info(self, model: torch.nn.Module):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Model Parameters:")
        self.logger.info(f"  Total: {total_params:,}")
        self.logger.info(f"  Trainable: {trainable_params:,}")

    def save_metrics(self):
        """Save metrics history to file."""
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        self.logger.info(f"Metrics saved to {metrics_file}")


# src/utils/ckpt.py
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        epoch: int,
        is_best: bool = False,
        metric: Optional[float] = None,
    ):
        """
        Save a checkpoint.

        Args:
            state: Dictionary containing model state and other info
            epoch: Current epoch
            is_best: Whether this is the best model so far
            metric: Metric value for this checkpoint
        """
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "level1_best.pt"
            torch.save(state, best_path)
            print(f"Saved best checkpoint to {best_path}")

        # Save regular checkpoint
        checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(state, checkpoint_path)

        # Track checkpoint
        self.checkpoints.append(
            {"path": checkpoint_path, "epoch": epoch, "metric": metric}
        )

        # Remove old checkpoints if exceeding max
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by metric (keep best) or by epoch (keep latest)
            if metric is not None:
                self.checkpoints.sort(
                    key=lambda x: x["metric"] if x["metric"] else float("inf")
                )
            else:
                self.checkpoints.sort(key=lambda x: x["epoch"])

            # Remove worst checkpoint
            to_remove = self.checkpoints[0]
            if to_remove["path"].exists():
                to_remove["path"].unlink()
                print(f"Removed old checkpoint: {to_remove['path']}")
            self.checkpoints.pop(0)

    def load_checkpoint(
        self, checkpoint_path: str, device: torch.device = torch.device("cpu")
    ):
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load the checkpoint to

        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from {checkpoint_path}")

        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if "val_losses" in checkpoint and "total" in checkpoint["val_losses"]:
            print(f"  Validation loss: {checkpoint['val_losses']['total']:.4f}")

        return checkpoint

    def get_best_checkpoint_path(self):
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "level1_best.pt"
        if best_path.exists():
            return str(best_path)

        # Fallback to latest checkpoint
        if self.checkpoints:
            return str(self.checkpoints[-1]["path"])

        # Check for any checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if checkpoint_files:
            return str(checkpoint_files[-1])

        return None


# src/utils/__init__.py
from .seed import set_seed
from .logging import MetricLogger
from .ckpt import CheckpointManager

__all__ = ["set_seed", "MetricLogger", "CheckpointManager"]
