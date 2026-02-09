"""Reproducibility utilities."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Random seed value.
        deterministic: If True, enforce deterministic CUDA operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
