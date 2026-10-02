"""Compare Gaia, CALSPEC, baseline, and PINN spectra in validation plots.

The module loads exported ``.npz`` archives and generates quantitative and
qualitative plots for individual stellar targets.

Notes
-----
Report context: Chapter 8, Model Evaluation and Visualization.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from src.utils.metrics.alternative_metrics import calculate_peak_normalized_mae


def generate_plots_from_file(
    input_filepath: str | Path, 
    output_dir: str | Path, 
    num_samples: int = 5,
    include_gaussian: bool = True,
) -> None:
    """
    Read inference results from a compressed `.npz` archive and generate validation plots.

    Extracts the wavelength grid, original uncalibrated spectra, reference spectra, 
    baseline predictions, and PINN predictions from the archive. For each target sample, 
    it generates a three-panel figure:
    1. Top Left: Absolute flux comparison across wavelength (shares X-axis with bottom left).
    2. Bottom Left: Percentage residual relative to the reference spectrum peak flux.
    3. Optional bottom right: Horizontal density histogram of residuals with a fitted theoretical Gaussian curve.

    Parameters
    ----------
    input_filepath : str | Path
        Full path to the `.npz` archive containing exported test predictions.
    output_dir : str | Path
        Destination directory where generated validation PNG figures will be saved.
        Created automatically if it does not exist.
    num_samples : int, default=5
        Maximum number of samples to plot starting from index 0. If the dataset 
        contains fewer samples than requested, all available samples are plotted.
    include_gaussian : bool, default=True
        Whether to include the residual histogram and fitted Gaussian panel.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the `.npz` file specified by `input_filepath` does not exist on disk.
    KeyError
        If mandatory data arrays (`wave`, `X_orig`, `Y_ref`, `X_pinn`) are missing from the archive.
    """
    input_path = Path(input_filepath)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found at: {input_path}")

    data = np.load(input_path)
    
    required_keys = {"wave", "X_orig", "Y_ref", "X_pinn"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise KeyError(f"Missing required arrays in `.npz` archive: {missing_keys}")

    wave = data["wave"]
    X_orig = data["X_orig"]
    Y_ref = data["Y_ref"]
    X_pinn = data["X_pinn"]
    X_huang = data["X_huang"] if "X_huang" in data.files else None

    output_path.mkdir(parents=True, exist_ok=True)
    samples_to_plot = min(num_samples, X_orig.shape[0])

    if wave.ndim != 1 or X_orig.ndim != 2 or Y_ref.ndim != 2 or X_pinn.ndim != 2:
        raise ValueError("wave must be 1D and spectral arrays must be 2D")
    if any(array.shape != Y_ref.shape for array in (X_orig, X_pinn)):
        raise ValueError("X_orig, Y_ref, and X_pinn must have the same shape")
    if wave.size != Y_ref.shape[1]:
        raise ValueError("wave length must match the spectral channel count")

    # Summarize residuals normalized by each reference spectrum's absolute peak.
    peak_flux = np.max(np.abs(Y_ref), axis=1, keepdims=True)

    def normalized_residuals(predictions: np.ndarray) -> np.ndarray:
        return np.divide(
            (predictions - Y_ref) * 100.0,
            peak_flux,
            out=np.zeros_like(predictions, dtype=float),
            where=peak_flux >= 1e-20,
        )

    residuals = {
        "Original Gaia": normalized_residuals(X_orig),
        "PINN": normalized_residuals(X_pinn),
    }
    if X_huang is not None and X_huang.shape == Y_ref.shape and np.any(X_huang):
        residuals["Huang baseline"] = normalized_residuals(X_huang)

    summary_fig, summary_ax = plt.subplots(figsize=(11, 5.5))
    colors = {"Original Gaia": "#4f88f3", "Huang baseline": "#00a83b", "PINN": "#e63946"}
    for label, residual in residuals.items():
        median = np.nanmedian(residual, axis=0)
        lower, upper = np.nanpercentile(residual, [25, 75], axis=0)
        summary_ax.fill_between(wave, lower, upper, color=colors[label], alpha=0.12)
        summary_ax.plot(wave, median, color=colors[label], linewidth=0.8, label=f"{label} median")

    summary_ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.8)
    # A reciprocal scale gives the shorter wavelengths more horizontal space.
    summary_ax.set_xscale("function", functions=(lambda x: -1.0 / (x + 1e-10), lambda x: -1.0 / (x + 1e-10)))
    summary_ax.set_xlim(336, 1020)
    summary_ax.set_xticks([336, 400, 500, 600, 700, 800, 900, 1020])
    summary_ax.set_xlabel("Wavelength [nm]", fontsize=18)
    summary_ax.set_ylabel("Peak-normalized residual (%)", fontsize=18)
    summary_ax.set_title("Median spectral residuals with interquartile range", fontsize=22)
    summary_ax.grid(True, linestyle="--", alpha=0.35)
    summary_ax.tick_params(axis="both", which="both", labelsize=12)
    summary_ax.legend(loc="best", fontsize=18)
    summary_fig.savefig(output_path / "wavelength_residuals.pdf", dpi=150, bbox_inches="tight")
    plt.close(summary_fig)

    boxplot_predictions = [X_orig, X_pinn]
    boxplot_labels = ["Original Gaia", "PINN"]
    boxplot_colors = ["#4f88f3", "#e63946"]
    if X_huang is not None and X_huang.shape == Y_ref.shape and np.any(X_huang):
        boxplot_predictions.insert(1, X_huang)
        boxplot_labels.insert(1, "Huang Baseline")
        boxplot_colors.insert(1, "#00a83b")

    pn_mae_values = [
        [calculate_peak_normalized_mae(Y_ref[i], predictions[i]) for i in range(Y_ref.shape[0])]
        for predictions in boxplot_predictions
    ]
    boxplot_fig, boxplot_ax = plt.subplots(figsize=(8, 5.5))
    boxplot = boxplot_ax.boxplot(
        pn_mae_values,
        tick_labels=boxplot_labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
    )
    for patch, color in zip(boxplot["boxes"], boxplot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.0)

    boxplot_ax.set_xlabel("Model", fontsize=18)
    boxplot_ax.set_ylabel("PN-MAE (%)", fontsize=18)
    boxplot_ax.set_title("Peak-normalized MAE by model", fontsize=22)
    boxplot_ax.tick_params(axis="both", which="both", labelsize=14)
    boxplot_ax.grid(axis="y", linestyle="--", alpha=0.35)
    boxplot_fig.savefig(output_path / "baseline_boxplot.pdf", dpi=150, bbox_inches="tight")
    plt.close(boxplot_fig)

    print(f"\n[>] Generating {samples_to_plot} figures from archive: {input_path}")

    for i in range(samples_to_plot):
        # Keep the histogram panel optional and let the spectral panels use the full width when omitted.
        fig = plt.figure(figsize=(14, 8))
        if include_gaussian:
            gs = fig.add_gridspec(
                2, 2,
                width_ratios=[2.5, 1],
                height_ratios=[2, 1],
                wspace=0.06,
                hspace=0.1,
            )
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
        else:
            gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.1)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        # Plot the absolute flux comparison in the top panel.
        ax1.plot(wave, Y_ref[i], label="CALSPEC Reference ($Y$)", color="black", alpha=0.85, linewidth=2)
        ax1.plot(wave, X_orig[i], label="Original Gaia BP/RP ($X$)", color="#4f88f3", alpha=0.6, linewidth=1.5)

        if X_huang is not None and np.any(X_huang[i] > 0):
            ax1.plot(
                wave, X_huang[i], 
                label="Huang et al. 2024 Baseline",
                color="#00cf49", alpha=0.9, linewidth=1.5, linestyle="--"
            )

        ax1.plot(wave, X_pinn[i], label=r"PINN Prediction ($X \cdot R$)", color="#e63946", alpha=0.8, linewidth=1.5)

        ax1.set_title(f"PINN vs Baseline | Star Index: {i:03d}", fontsize=24)
        ax1.set_ylabel(r"Flux ($\mathrm{W\,m^{-2}\,nm^{-1}}$)", fontsize=20)
        ax1.tick_params(axis="both", which="both", labelsize=16)
        ax1.yaxis.get_offset_text().set_fontsize(16)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=18)

        # Hide top-panel x-axis labels to prevent overlap with the lower panel.
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Plot percentage residuals in the lower-left panel.
        res_original_pct = residuals["Original Gaia"][i]
        res_pinn_pct = residuals["PINN"][i]

        ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.85)
        ax2.plot(wave, res_original_pct, label="Original Gaia Residual", color="#4f88f3", alpha=0.75, linewidth=1.0)

        if X_huang is not None and np.any(X_huang[i] != 0):
            res_baseline_pct = residuals["Huang baseline"][i]
            ax2.plot(
                wave, res_baseline_pct, 
                label="Huang Baseline Residual", 
                color="#00cf49", alpha=0.8, linewidth=1.5, linestyle="--"
            )

        ax2.plot(wave, res_pinn_pct, label="PINN Residual", color="#e63946", alpha=0.9, linewidth=1.5)

        ax2.set_xlabel("Wavelength [nm]", fontsize=20)
        ax2.set_ylabel("Residual (%)", fontsize=20)
        ax2.tick_params(axis="both", which="both", labelsize=16)
        ax2.set_ylim(-20, 20)
        ax2.grid(True, linestyle="--", alpha=0.4)

        if include_gaussian:
            # Plot the residual histogram and Gaussian fit in the lower-right panel.
            bins = np.linspace(-20, 20, 40)

            ax3.hist(res_original_pct, bins=bins, orientation="horizontal", color="#4f88f3", alpha=0.3, density=True)
            if X_huang is not None and np.any(X_huang[i] != 0):
                ax3.hist(res_baseline_pct, bins=bins, orientation="horizontal", color="#00cf49", alpha=0.3, density=True)
            ax3.hist(res_pinn_pct, bins=bins, orientation="horizontal", color="#e63946", alpha=0.6, density=True)

            mu, std = norm.fit(res_pinn_pct)
            y_vals = np.linspace(-20, 20, 100)
            p = norm.pdf(y_vals, mu, std)
            ax3.plot(p, y_vals, "k--", linewidth=1.5, label=rf"Fit: $\mu$={mu:.1f}, $\sigma$={std:.1f}")

            ax3.axhline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.85)
            ax3.set_xlabel("Density", fontsize=20)
            ax3.tick_params(axis="both", which="both", labelsize=16, labelleft=False)
            ax3.grid(True, linestyle="--", alpha=0.4)
            ax3.legend(fontsize=14, loc="upper right")

        # Save the figure.
        save_file = output_path / f"star_comparison_test_{i:03d}.pdf"
        plt.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[✓] Visualization complete. {samples_to_plot} plots saved to: {output_path}")


def plot_training_history(input_filepath: str | Path, output_filepath: str | Path) -> None:
    """Plot LOOCV training and held-out validation losses for both phases."""
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)
    if not input_path.exists():
        raise FileNotFoundError(f"Training history not found at: {input_path}")

    data = np.load(input_path)
    required_keys = {"base_train", "base_val", "blue_train", "blue_val"}
    missing_keys = required_keys - set(data.files)
    if missing_keys:
        raise KeyError(f"Missing training history arrays: {missing_keys}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    panels = (
        (axes[0], data["base_train"], data["base_val"], "Base PINN (Phase 1)"),
        (axes[1], data["blue_train"], data["blue_val"], "BlueWiggleNet (Phase 2)"),
    )
    for axis, train_history, val_history, title in panels:
        histories = (
            (train_history, "Training", "#4f88f3"),
            (val_history, "Validation", "#e63946"),
        )
        summaries = []
        epoch_limit = 1
        for history, label, color in histories:
            history = np.atleast_2d(history)
            if not history.size:
                continue
            median = np.nanmedian(history, axis=0)
            low, high = np.nanpercentile(history, [25, 75], axis=0)
            epochs = np.arange(1, median.size + 1)
            epoch_limit = max(epoch_limit, median.size)
            summaries.append((epochs, median, label, color))
            axis.fill_between(epochs, low, high, color=color, alpha=0.12)
        for epochs, median, label, color in summaries:
            axis.plot(epochs, np.clip(median, 1e-16, None), color=color,
                      linewidth=0.9, label=label)
        axis.set_xlim(0, epoch_limit)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch", fontsize=18)
        axis.set_ylabel("Loss", fontsize=18)
        axis.set_title(title, fontsize=22)
        axis.grid(True, which="both", linestyle="--", alpha=0.3)
        axis.tick_params(axis="both", which="both", labelsize=14)
        axis.legend(fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_plots_from_file(
        input_filepath="results/test_predictions_final.npz", 
        output_dir="plots/validation_figures",
        num_samples=5
    )