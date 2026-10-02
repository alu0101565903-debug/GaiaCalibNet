"""Plot representative residuals across stellar magnitude and color bins.

The module loads exported predictions and metadata, selects representative
spectra, and writes a multi-panel residual figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np


DEFAULT_ARCHIVE = Path("results/models/loocv_eps_residual_pinn/loocv_aggregated_predictions.npz")
DEFAULT_DATA_DIR = Path("data/processed/nn_data")
DEFAULT_OUTPUT = Path("results/models/loocv_eps_residual_pinn/residual_grid_multipanel.pdf")

COLOR_EDGES = np.array([-np.inf, 0.3, np.inf])


def _load_color(data_dir: Path, expected_size: int) -> np.ndarray:
    """Load and validate BP-RP color metadata.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory containing the metadata arrays.
    expected_size : int
        Required number of color values.

    Returns
    -------
    numpy.ndarray
        Flattened BP-RP color values.

    Raises
    ------
    FileNotFoundError
        If the color metadata file does not exist.
    ValueError
        If the number of values differs from ``expected_size``.
    """
    color_path = data_dir / "bp_rp.npy"
    if not color_path.exists():
        raise FileNotFoundError(
            f"Could not find BP-RP metadata at {color_path}. "
            "Pass --data-dir containing bp_rp.npy."
        )

    color = np.asarray(np.load(color_path), dtype=float).reshape(-1)
    if color.size != expected_size:
        raise ValueError(f"BP-RP length does not match sample count.")
    return color


def _relative_residual(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Calculate residuals as a percentage of the reference peak flux.

    Parameters
    ----------
    prediction : numpy.ndarray
        Predicted flux values.
    reference : numpy.ndarray
        Reference flux values.

    Returns
    -------
    numpy.ndarray
        Peak-normalized residual values in percent.
    """
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    
    peak_flux = np.max(reference)
    if peak_flux < 1e-18:
        peak_flux = 1e-16
        
    return 100.0 * (prediction - reference) / peak_flux


def _representative_index(
    indices: np.ndarray, raw_array: np.ndarray, ref_array: np.ndarray, wave: np.ndarray
) -> int:
    """Select the sample with the largest mean absolute normalized residual.

    Parameters
    ----------
    indices : numpy.ndarray
        Candidate sample indices.
    raw_array : numpy.ndarray
        Uncalibrated flux spectra.
    ref_array : numpy.ndarray
        Reference flux spectra.
    wave : numpy.ndarray
        Wavelength grid in nanometers.

    Returns
    -------
    int
        Index of the selected sample.
    """
    if indices.size == 1:
        return int(indices[0])

    errors = []
    for i in indices:
        peak_flux = np.max(ref_array[i])
        if peak_flux < 1e-18:
            peak_flux = 1e-16

        eval_mask = (wave > 500) & (wave < 950) & (ref_array[i] > (0.05 * peak_flux))
        res = (raw_array[i][eval_mask] - ref_array[i][eval_mask]) / peak_flux
        errors.append(np.nanmean(np.abs(res)))
    
    return int(indices[np.argmax(errors)])


def plot_residual_grid(
    input_filepath: str | Path = DEFAULT_ARCHIVE,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = DEFAULT_OUTPUT,
    color_edges: Sequence[float] = COLOR_EDGES,
    dpi: int = 300,
) -> Path:
    """Create and save a binned multi-panel residual plot.

    Parameters
    ----------
    input_filepath : str or pathlib.Path, optional
        Path to the archive containing prediction arrays.
    data_dir : str or pathlib.Path, optional
        Directory containing fallback wavelength and stellar metadata arrays.
    output_path : str or pathlib.Path, optional
        Destination path for the output figure.
    color_edges : sequence of float, optional
        BP-RP color-bin edges.
    dpi : int, optional
        Resolution used when saving the figure.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    input_filepath = Path(input_filepath)
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    color_edges = np.asarray(color_edges, dtype=float)

    with np.load(input_filepath, allow_pickle=False) as archive:
        # Fall back to data_dir when arrays are missing from the archive.
        wave_raw = archive.get("wave")
        if wave_raw is None:
            wave_raw = np.load(data_dir / "wave_grid.npy")
        wave = np.asarray(wave_raw, dtype=float).reshape(-1)
        
        raw = np.asarray(archive.get("X_orig", archive.get("X")), dtype=float)
        reference = np.asarray(archive.get("Y_ref", archive.get("Y_true")), dtype=float)
        
        # Load the empirical baseline when available.
        empirical_raw = archive.get("X_huang")
        has_empirical = empirical_raw is not None
        if has_empirical:
            empirical = np.asarray(empirical_raw, dtype=float)
        
        g_mags_raw = archive.get("G_mags")
        if g_mags_raw is None:
            g_mags_raw = np.load(data_dir / "G_mags.npy")
        g_mags = np.asarray(g_mags_raw, dtype=float).reshape(-1)
        
        # LOOCV stores the ratio in Y_pred; reconstruct the PINN flux as X * ratio.
        pinn = np.asarray(archive.get("Y_pred", archive.get("X_pinn")), dtype=float)

    sample_count = g_mags.size
    
    try:
        colors = _load_color(data_dir, sample_count)
    except Exception as e:
        print(f"Warning: {e}. Falling back to standardized stellar parameters.")
        params = np.asarray(archive.get("P", archive.get("params")), dtype=float)
        colors = params[:, 0]

    p33 = np.nanpercentile(g_mags, 33.33)
    p67 = np.nanpercentile(g_mags, 66.67)
    mag_edges = np.array([-np.inf, p33, p67, np.inf])

    row_count = mag_edges.size - 1
    column_count = color_edges.size - 1
    fig, axes = plt.subplots(
        row_count, column_count,
        figsize=(3.15 * column_count, 2.15 * row_count),
        sharex=True, sharey=True, squeeze=False,
    )
    
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xticks([400, 500, 600, 800, 1000], labels=["400", "500", "600", "800", "1000"])
    axes[0, 0].xaxis.set_minor_formatter(NullFormatter())

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.08, top=0.92, hspace=0.08, wspace=0.08)

    y_limit = 15.0
    handles, labels = [], []
    
    for row in range(row_count):
        mag_mask = (g_mags >= mag_edges[row]) & (g_mags < mag_edges[row + 1])
        for column in range(column_count):
            axis = axes[row, column]
            cell_mask = mag_mask & (colors >= color_edges[column]) & (colors < color_edges[column + 1])
            indices = np.flatnonzero(cell_mask & np.isfinite(g_mags) & np.isfinite(colors))

            if indices.size:
                index = _representative_index(indices, raw, reference, wave)
                
                # Build the curves available for this sample.
                curves_data = [
                    ("Uncalibrated Gaia", "#2878b5", "-", 0.4, raw[index]),
                ]
                if has_empirical:
                    curves_data.append(("Empirical baseline", "#0a5825", "--", 0.4, empirical[index]))
                curves_data.append(("PINN", "#d62728", "-", 0.4, pinn[index]))

                axis.axvline(450.0, color="#888888", linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)

                for curve_number, (label, color, linestyle, linewidth, pred_data) in enumerate(curves_data):
                    residual = _relative_residual(pred_data, reference[index])
                    line, = axis.plot(wave, residual, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
                    
                    if row == 0 and column == 0:
                        handles.append(line)
                        labels.append(label)
                
                axis.text(
                    0.04, 0.91, f"N={indices.size} | Idx: {index:03d}", transform=axis.transAxes,
                    fontsize=8, color="#222222", va="top",
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)
                )

                axis.axhline(0.0, color="#555555", linewidth=0.55, alpha=0.65)
                axis.set_ylim(-y_limit, y_limit)
                axis.grid(True, linewidth=0.35, alpha=0.3)
                axis.tick_params(labelsize=7)
                
                if row == 0:
                    color_label = f"BP-RP < {color_edges[column + 1]:g}" if not np.isfinite(color_edges[column]) else \
                                  f"BP-RP $\\geq$ {color_edges[column]:g}" if not np.isfinite(color_edges[column + 1]) else \
                                  f"{color_edges[column]:g} $\\leq$ BP-RP < {color_edges[column + 1]:g}"
                    axis.set_title(color_label, fontsize=9)
                    
                if column == 0:
                    magnitude_label = f"G < {mag_edges[row + 1]:.1f} mag" if not np.isfinite(mag_edges[row]) else \
                                      f"G $\\geq$ {mag_edges[row]:.1f} mag" if not np.isfinite(mag_edges[row + 1]) else \
                                      f"{mag_edges[row]:.1f} $\\leq$ G < {mag_edges[row + 1]:.1f} mag"
                    axis.set_ylabel(f"{magnitude_label}\nResidual (%)", fontsize=8)
            else:
                axis.set_visible(False)

    for column in range(column_count):
        for row in range(row_count - 1, -1, -1):
            ax = axes[row, column]
            if ax.get_visible():
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.set_xlabel("Wavelength (nm)", fontsize=8)
                ax.set_xlim(390, 1000)
                break

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, fontsize=9)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Residual grid saved to: {output_path}")
    return output_path


def main() -> None:
    """Parse command-line arguments and generate the residual plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ARCHIVE, help="Aggregated prediction archive (.npz).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing bp_rp.npy.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    plot_residual_grid(args.input, args.data_dir, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()