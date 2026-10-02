"""Configure reproducible random state and select compute devices.

The utilities seed Python, NumPy, and PyTorch random-number generators and
configure deterministic backend behavior for PINN training.
"""

import os
import random
import numpy as np
import torch
from tqdm.auto import tqdm


def set_seed(seed: int = 42, *, verbose: bool = True):
    """
    Set random seeds for Python, NumPy, PyTorch, and CUDA.

    Configure deterministic backend behavior for repeatable model training.

    Parameters
    ----------
    seed : int, optional
        Random seed to apply. The default is 42.
    verbose : bool, optional
        If True, report the seed using the progress-bar-aware writer. Set to
        False for repeated per-fold resets. The default is True.

    Returns
    -------
    None

    Notes
    -----
    This function modifies global random-number-generator and backend state.
    """
    # Configure these before initializing CUDA operations.
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Seed Python and NumPy.
    random.seed(seed)
    np.random.seed(seed)
    
    # Seed PyTorch for CPU and multi-GPU execution.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force deterministic cuDNN algorithms and disable auto-tuning.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Warn when an operation has no deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if verbose:
        tqdm.write(f"-> Global random seed strictly locked to: {seed}")


def get_device() -> torch.device:
    """
    Return the best available compute device.

    Returns
    -------
    torch.device
        CUDA when available; otherwise, CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")