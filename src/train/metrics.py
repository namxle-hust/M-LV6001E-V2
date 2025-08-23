# src/train/metrics.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Optional


class MetricTracker:
    """Track and compute metrics during training."""

    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ["loss", "mse", "auroc"]
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.values = {metric: [] for metric in self.metrics}
        self.counts = {metric: 0 for metric in self.metrics}

    def update(self, metric: str, value: float, n: int = 1):
        """Update a metric with a new value."""
        if metric not in self.metrics:
            self.metrics.append(metric)
            self.values[metric] = []
            self.counts[metric] = 0

        self.values[metric].append(value * n)
        self.counts[metric] += n

    def get_average(self, metric: str) -> float:
        """Get average value for a metric."""
        if metric not in self.values or not self.values[metric]:
            return 0.0

        total = sum(self.values[metric])
        count = self.counts[metric]

        return total / count if count > 0 else 0.0

    def get_all_averages(self) -> Dict[str, float]:
        """Get all average values."""
        return {metric: self.get_average(metric) for metric in self.metrics}


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, task: str = "binary"
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted values
        task: Type of task ('binary', 'regression')

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    if task == "binary":
        # Binary classification metrics
        if len(np.unique(y_true)) > 1:  # Check if both classes present
            metrics["auroc"] = roc_auc_score(y_true, y_pred)
            metrics["auprc"] = average_precision_score(y_true, y_pred)

        # Threshold-based metrics
        y_pred_binary = (y_pred > 0.5).astype(int)
        metrics["accuracy"] = np.mean(y_true == y_pred_binary)

        # Precision, recall, F1
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

    elif task == "regression":
        # Regression metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)

        # R-squared
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics["mse"] = mse
        metrics["mae"] = mae
        metrics["rmse"] = rmse
        metrics["r2"] = r2

    return metrics


# src/train/scheduler.py
import torch.optim as optim
from typing import Optional


def get_scheduler(
    optimizer: optim.Optimizer, scheduler_type: str, **kwargs
) -> Optional[object]:
    """
    Get learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        **kwargs: Additional scheduler parameters

    Returns:
        Learning rate scheduler or None
    """
    if scheduler_type == "none":
        return None

    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 50),
            gamma=kwargs.get("gamma", 0.5),
        )

    elif scheduler_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get("gamma", 0.95)
        )

    elif scheduler_type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
            min_lr=kwargs.get("min_lr", 1e-6),
        )

    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 1e-6),
        )

    elif scheduler_type == "cosine_warm_restarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("T_0", 50),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("eta_min", 1e-6),
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
