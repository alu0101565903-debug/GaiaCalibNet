"""
Module: inspect_data.py

Description:
    Loads a .npy file and prints its full contents without truncation 
    for debugging large datasets.
"""
import numpy as np
import sys
from pathlib import Path

def print_full_npy(file_path: str | Path):
    """Loads a .npy file and prints its entire content without truncation."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File not found at '{path}'")
        return
        
    try:
        data = np.load(path)
        np.set_printoptions(threshold=sys.maxsize, linewidth=200)
        
        print("=" * 50)
        print(f"FULL CONTENT OF: {path.name}")
        print("=" * 50)
        print(data)
        print("=" * 50)
        
    except Exception as e:
        print(f"Error reading the file: {e}")

if __name__ == "__main__":
    print_full_npy("data/processed/wave_grid.npy")