"""
K-fold cross-validation utilities for patient splitting.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import KFold, StratifiedKFold


def create_kfold_splits(
    n_samples: int, n_folds: int = 5, seed: int = 42, stratify_labels: np.ndarray = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold splits for cross-validation.

    Args:
        n_samples: Total number of samples
        n_folds: Number of folds
        seed: Random seed
        stratify_labels: Optional labels for stratified splitting

    Returns:
        List of (train_indices, val_indices) tuples
    """
    indices = np.arange(n_samples)

    if stratify_labels is not None:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(indices, stratify_labels))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(indices))

    return splits
