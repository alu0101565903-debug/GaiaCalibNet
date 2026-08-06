"""
Module: plot_loss.py

Plot the training loss history for a physics-informed neural network (PINN).
This module creates a line plot showing the total loss and its individual
loss components (for example, flux loss and smoothing loss) across epochs.
"""

import os
from typing import Dict, Iterable

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_evolution(loss_history: Dict[str, Iterable[float]], output_filepath: str) -> None:
    """
    Plot the evolution of loss terms recorded during training.

    This function saves a figure containing one line per loss term. The y-axis
    uses a logarithmic scale so that both the total loss and smaller component
    losses remain visible over many epochs.

    Args:
    - loss_history: Mapping from each loss term name to its values over epochs.
        Example keys are "total_loss", "loss_flux", "loss_smooth", "loss_reg".
    - output_filepath: Full path where the saved PNG figure will be written.

    Notes:
        - If the target directory does not already exist, it is created automatically.
        - Any loss term that is not recognized in the default color palette will
          use Matplotlib's default line colors.
    """
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not loss_history:
        raise ValueError("loss_history must contain at least one loss term.")

    plt.figure(figsize=(10, 6))

    # Default colors for common loss terms.
    colors = {
        "total_loss": "black",
        "loss_flux": "#4f88f3",
        "loss_smooth": "#00cf49",
        "loss_reg": "#e63946",
    }

    for loss_name, loss_values in loss_history.items():
        color = colors.get(loss_name, None)
        line_width = 2.5 if loss_name == "total_loss" else 1.5
        line_style = "-" if loss_name == "total_loss" else "--"
        alpha = 0.9 if loss_name == "total_loss" else 0.75

        plt.plot(
            loss_values,
            label=loss_name.replace("_", " ").title(),
            color=color,
            linewidth=line_width,
            linestyle=line_style,
            alpha=alpha,
        )

    plt.title("Loss Evolution During Training", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize="medium")
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[✓] Saved loss evolution plot to: {output_filepath}")


if __name__ == "__main__":
    # Example usage with synthetic loss data. In production, import "plot_loss_evolution"
    # from this module and pass real training history.
    epochs = 1000
    mock_epochs = np.arange(epochs)

    # Create mock curves that resemble a decaying training loss.
    mock_flux = np.exp(-mock_epochs / 200) + 0.01
    mock_smooth = (np.exp(-mock_epochs / 300) + 0.05) * 0.1
    mock_total = mock_flux + mock_smooth

    dummy_history = {
        "total_loss": mock_total,
        "loss_flux": mock_flux,
        "loss_smooth": mock_smooth,
    }

    plot_loss_evolution(dummy_history, "results/plots/loss_evolution_demo.png")