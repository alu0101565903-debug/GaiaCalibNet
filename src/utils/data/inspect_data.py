"""
Module: inspect_data.py

Loads a .npy binary file and prints its complete array contents without 
truncation for dataset inspection and debugging.
"""

from pathlib import Path
import sys
import numpy as np


def print_full_npy(file_path: str | Path) -> None:
    """
    Load a NumPy `.npy` file and print its entire content without truncation.

    Temporarily modifies NumPy array printing options using a context manager
    to display all array elements and prevent inline truncation (`...`).

    Parameters
    ----------
    file_path : str | Path
        Filepath pointing to the target `.npy` binary array file.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist on disk.
    RuntimeError
        If an error occurs while deserializing or reading the `.npy` file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Target file not found at: '{path}'")

    try:
        data = np.load(path)
        
        with np.printoptions(threshold=sys.maxsize, linewidth=200):
            print("=" * 50)
            print(f"FULL CONTENT OF: {path.name}")
            print("=" * 50)
            print(data)
            print("=" * 50)

    except Exception as e:
        raise RuntimeError(f"Error reading file '{path}': {e}") from e


if __name__ == "__main__":
    print_full_npy("data/processed/nn/all/wave_grid.npy")