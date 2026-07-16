"""
Module: plot_raw_data.py
Report Section: Chapter 4 / Chapter 5 (Data Diagnostics & Preprocessing)

This module generates diagnostic comparison plots of raw Gaia XP input spectra
against their CALSPEC target spectra. It is designed for preprocessing quality
control and saves publication-quality figures directly to disk without opening
interactive windows.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_stars(out_dir: str = "results/plots/diagnostics"):
    """
    Load preprocessed spectral matrices and generate side-by-side diagnostic
    comparison plots for two sample stars.

    The function compares the raw input spectrum X with the target spectrum Y
    for the first two available stars, then saves the resulting figure to the
    configured output directory.

    Parameters:
    - out_dir (str): directory where the plot image will be saved. Defaults to
      "results/plots/diagnostics".
    """
    print("=" * 60)
    print("[DATA DIAGNOSTICS] GENERATING PREPROCESSING COMPARISON PLOTS")
    print("=" * 60)

    x_path = "data/processed/X.npy"
    y_path = "data/processed/Y.npy"

    # Verify that dataset matrices exist before attempting to load them.
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"[!] ERROR: Processed matrices not found at: data/processed/")
        return

    # Load the preprocessed input and target spectral matrices.
    print("-> Loading preprocessed dataset arrays...")
    X = np.load(x_path)
    Y = np.load(y_path)

    n_samples = X.shape[0]
    if n_samples == 0:
        print("[!] ERROR: Loaded X.npy is empty.")
        return

    # Choose up to two stars for comparison. If only one star exists, reuse it.
    idx1 = 0
    idx2 = 1 if n_samples > 1 else 0

    # Configure a 1x2 side-by-side subplot layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- PANEL 1: Diagnostic Star 1 ---
    print(f"-> Plotting diagnostic Star 1 (Index {idx1})...")
    ax1.plot(X[idx1], label="Uncalibrated Input (X)", color="#264ab7", alpha=0.7, linewidth=1.5)
    ax1.plot(Y[idx1], label="Target Ground Truth (Y)", color="#d62728", alpha=0.7, linewidth=1.5)
    ax1.set_title(f"Diagnostic Sample Star 1 (Index {idx1})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Wavelength Channel / Pixel Index (0-300)", fontsize=11)
    ax1.set_ylabel("Flux Intensity", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- PANEL 2: Diagnostic Star 2 ---
    print(f"-> Plotting diagnostic Star 2 (Index {idx2})...")
    ax2.plot(X[idx2], label="Uncalibrated Input (X)", color="#264ab7", alpha=0.7, linewidth=1.5)
    ax2.plot(Y[idx2], label="Target Ground Truth (Y)", color="#d62728", alpha=0.7, linewidth=1.5)
    ax2.set_title(f"Diagnostic Sample Star 2 (Index {idx2})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Wavelength Channel / Pixel Index (0-300)", fontsize=11)
    ax2.set_ylabel("Flux Intensity", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # --- SAVE FILE (no interactive display) ---
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "raw_vs_target_spectral_samples.png")
    
    # Save with high-resolution dpi=300 for publication-quality output.
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Free memory after writing the figure to disk.
    
    print("-" * 60)
    print(f"[+] Diagnostic plot successfully saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    out_dir = "results/plots/diagnostics"
    plot_sample_stars(out_dir)