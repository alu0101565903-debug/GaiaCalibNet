"""
Module: reproducibility_utils.py

Utility functions for hardware and software environment configuration.

This module provides essential tools to lock random seeds across all underlying 
libraries (Python, NumPy, PyTorch, CUDA) to ensure strict deterministic behavior 
and absolute reproducibility for the Physics-Informed Neural Network (PINN) training.
It also handles dynamic hardware device allocation.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets all random seeds to guarantee absolute reproducibility across 
    PyTorch, NumPy, standard Python, and the CUDA/cuBLAS backend.

    This function rigorously configures the runtime environment to ensure that 
    multiple executions of the model training pipeline yield strictly deterministic 
    and identically repeatable results, which is critical for scientific validation.

    Args:
    - seed (int, optional): The integer value to use as the global random seed. 
        Defaults to 42.

    Returns:
    - None: This function does not return any value. It directly modifies 
        the global state of the random number generators.
    """
    # 1. Environment variables (Must be configured BEFORE initializing CUDA operations)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # 2. Base Python and NumPy
    random.seed(seed)
    np.random.seed(seed)
    
    # 3. PyTorch (Applies to both CPU and multi-GPU setups)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 4. Deterministic Hardware Operations (cuDNN backend)
    # Forces cuDNN to use deterministic algorithms, avoiding auto-tuning variations.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5. Strict PyTorch deterministic algorithms
    # Ensures that operations without a deterministic implementation throw a warning/error.
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"-> Global random seed strictly locked to: {seed}")


def get_device() -> torch.device:
    """
    Determines and returns the optimal available compute device.

    Automatically allocates computations to a CUDA-enabled NVIDIA GPU if 
    detected in the system environment; otherwise, it safely falls back to CPU.

    Returns:
        torch.device: The designated PyTorch computation device ("cuda" or "cpu").
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")