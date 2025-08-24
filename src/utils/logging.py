"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd
from datetime import datetime


def setup_logging(
    log_dir: str, log_level: str = "INFO", log_file: str = "train.log"
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_file: Name of log file

    Returns:
        Logger instance
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = logging.getLogger("multimodal_gnn")
    logger.setLevel(getattr(logging, log_level))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(Path(log_dir) / log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_metrics(
    metrics: Dict[str, Any],
    epoch: int,
    stage: str,
    logger: logging.Logger = None,
    writer=None,
):
    """
    Log metrics to console, file, and tensorboard.

    Args:
        metrics: Dictionary of metrics
        epoch: Current epoch
        stage: Training stage ('train', 'val', 'test')
        logger: Logger instance
        writer: Tensorboard writer
    """
    # Format metrics string
    metrics_str = f"Epoch {epoch} [{stage}]: "
    metrics_str += ", ".join(
        [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ]
    )

    # Log to console/file
    if logger:
        logger.info(metrics_str)
    else:
        print(metrics_str)

    # Log to tensorboard
    if writer:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{stage}/{key}", value, epoch)


class MetricsTracker:
    """Track and save training metrics."""

    def __init__(self, save_dir: str):
        """
        Initialize metrics tracker.

        Args:
            save_dir: Directory to save metrics
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = {"train": [], "val": [], "test": []}

    def update(self, metrics: Dict, epoch: int, stage: str):
        """
        Update metrics history.

        Args:
            metrics: Metrics dictionary
            epoch: Current epoch
            stage: Stage ('train', 'val', 'test')
        """
        metrics_with_epoch = {"epoch": epoch, **metrics}
        self.metrics_history[stage].append(metrics_with_epoch)

    def save(self):
        """Save metrics to files."""
        # Save as JSON
        with open(self.save_dir / "metrics_history.json", "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Save as CSV
        for stage, history in self.metrics_history.items():
            if history:
                df = pd.DataFrame(history)
                df.to_csv(self.save_dir / f"{stage}_metrics.csv", index=False)

    def get_best_epoch(
        self, metric: str = "loss", stage: str = "val", mode: str = "min"
    ):
        """
        Get best epoch based on metric.

        Args:
            metric: Metric to use
            stage: Stage to check
            mode: 'min' or 'max'

        Returns:
            Best epoch number and metric value
        """
        if not self.metrics_history[stage]:
            return None, None

        df = pd.DataFrame(self.metrics_history[stage])

        if mode == "min":
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()

        best_epoch = df.loc[best_idx, "epoch"]
        best_value = df.loc[best_idx, metric]

        return best_epoch, best_value


def create_experiment_summary(config: dict, metrics: dict, save_path: str):
    """
    Create experiment summary report.

    Args:
        config: Configuration dictionary
        metrics: Final metrics
        save_path: Path to save summary
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "final_metrics": metrics,
        "model_architecture": {
            "encoder_type": config["model"]["encoder"]["type"],
            "hidden_size": config["model"]["encoder"]["hidden_size"],
            "num_layers": config["model"]["encoder"]["num_layers"],
            "pooling_type": config["model"]["pooling"]["type"],
        },
        "training": {
            "stage_a_epochs": config["training"]["stage_a"]["epochs"],
            "stage_b_epochs": config["training"]["stage_b"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rates": {
                "stage_a": config["training"]["stage_a"]["learning_rate"],
                "stage_b": config["training"]["stage_b"]["learning_rate"],
            },
        },
    }

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Experiment summary saved to {save_path}")
