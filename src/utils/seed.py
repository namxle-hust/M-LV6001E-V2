"""
Utilities for reproducibility and seed management.
"""

import torch
import numpy as np
import random
import os


def set_seed(seed: int, cuda_deterministic: bool = False):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
        cuda_deterministic: Whether to use deterministic CUDA operations (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster but less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_random_state():
    """Get current random state for all random number generators."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def set_random_state(state: dict):
    """Restore random state for all random number generators."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
