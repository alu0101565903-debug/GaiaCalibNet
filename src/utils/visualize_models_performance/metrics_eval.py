"""Evaluate exported spectral predictions with astrophysical metrics.

The module computes per-target accuracy, uncertainty, bias, and similarity
metrics, then exports aggregate results as a LaTeX table and PNG image.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.metrics.alternative_metrics import (
    calculate_peak_normalized_mae,
    calculate_normalized_std_percentage,
    calculate_reduced_chi_square,
    calculate_normalized_bias_percentage,
    calculate_cosine_similarity,
    calculate_band_errors,
    calculate_band_bias
)

def evaluate_test_set_from_file(input_filepath: str, output_dir: str, num_knots: int = 20, model_name: str = "PINN"):
    """
    Evaluate exported predictions with astrophysical metrics.

    Calculate metrics for each stellar sample, then export their means as a
    LaTeX table and PNG image.

    Parameters
    ----------
    input_filepath : str
        Path to the ``.npz`` archive containing spectra, references, predictions,
        and uncertainties.
    output_dir : str
        Destination directory for the generated ``.tex`` and ``.png`` files.
    num_knots : int, optional
        Number of model parameters used to calculate reduced chi-square degrees
        of freedom. The default is 20.
    model_name : str, optional
        Model label used in the generated table. The default is ``"PINN"``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If ``input_filepath`` does not exist.
    """
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"File not found: {input_filepath}")
        
    # Load the exported arrays.
    data = np.load(input_filepath)
    wave = data["wave"]
    y_true = data["Y_ref"]
    y_pred = data["X_pinn"]

    # Robust error handling: if errors were not saved in the .npz, use a conservative epsilon
    if "X_err" in data:
        y_err = data["X_err"]
    else:
        y_err = np.ones_like(y_true) * 1e-4

    num_stars = y_true.shape[0]
    
    # Initialize metric accumulators.
    pn_mae_list, std_list, chi_sq_list, bias_list, cos_sim_list = [], [], [], [], []
    band_mae_values = {"blue": [], "green": [], "red": []}
    band_bias_values = {"blue": [], "green": [], "red": []}
    
    # Calculate metrics for each star.
    for i in range(num_stars):
        pn_mae_list.append(calculate_peak_normalized_mae(y_true[i], y_pred[i]))
        std_list.append(calculate_normalized_std_percentage(y_true[i], y_pred[i]))
        chi_sq_list.append(calculate_reduced_chi_square(y_true[i], y_pred[i], y_err[i], n_params=num_knots))
        bias_list.append(calculate_normalized_bias_percentage(y_true[i], y_pred[i]))
        cos_sim_list.append(calculate_cosine_similarity(y_true[i], y_pred[i]))
        
        # Calculate band-specific errors.
        b_mae = calculate_band_errors(wave, y_true[i], y_pred[i])
        band_mae_values["blue"].append(b_mae["blue_rel_error_percent"])
        band_mae_values["green"].append(b_mae["green_rel_error_percent"])
        band_mae_values["red"].append(b_mae["red_rel_error_percent"])

        # Calculate band-specific bias.
        b_bias = calculate_band_bias(wave, y_true[i], y_pred[i])
        band_bias_values["blue"].append(b_bias["blue_bias_norm_percent"])
        band_bias_values["green"].append(b_bias["green_bias_norm_percent"])
        band_bias_values["red"].append(b_bias["red_bias_norm_percent"])
        
    # Average each metric across stars.
    avg_pn_mae = np.mean(pn_mae_list)
    avg_std = np.mean(std_list)
    avg_chi_sq = np.mean(chi_sq_list)
    avg_bias = np.mean(bias_list)
    avg_cos_sim = np.mean(cos_sim_list)

    avg_blue_mae = np.mean(band_mae_values["blue"])
    avg_green_mae = np.mean(band_mae_values["green"])
    avg_red_mae = np.mean(band_mae_values["red"])

    avg_blue_bias = np.mean(band_bias_values["blue"])
    avg_green_bias = np.mean(band_bias_values["green"])
    avg_red_bias = np.mean(band_bias_values["red"])

    # Generate the LaTeX table.
    latex_table = f"""
\\begin{{table}}[htpb]
\\centering
\\caption{{Quantitative Evaluation Results of the {model_name} Model (Evaluation sample: {num_stars} stars)}}
\\label{{tab:{model_name.lower()}_evaluation}}
\\begin{{tabular}}{{lcll}}
\\hline
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Objective}} & \\textbf{{Utility}} \\\\
\\hline
Peak-Normalized MAE & {avg_pn_mae:.3f}\\% & 0.0\\% & Global precision independent of magnitude \\\\
Normalized Std. Dev. (NSTD) & {avg_std:.3f}\\% & 0.0\\% & Statistical dispersion of flux residuals \\\\
Reduced $\\chi^2_\\nu$ & {avg_chi_sq:.3f} & $\\approx 1.0$ & Statistical validity relative to noise $\\sigma_i$ \\\\
Normalized Bias (MBE) & {avg_bias:.3f}\\% & 0.0\\% & Integrated photometric flux conservation \\\\
Cosine Similarity & {avg_cos_sim:.5f} & 1.00000 & Macroscopic continuum morphological fidelity \\\\
\\hline
\\multicolumn{{4}}{{c}}{{\\textbf{{Relative Error by Band}}}} \\\\
\\hline
Blue Band (< 500 nm) & {avg_blue_mae:.3f}\\% & Minimum & Regional response in Gaia BP photometer \\\\
Green Band (500 - 700 nm) & {avg_green_mae:.3f}\\% & Minimum & Regional response in central optical band \\\\
Red Band (> 700 nm) & {avg_red_mae:.3f}\\% & Minimum & Regional response in Gaia RP photometer \\\\
\\hline
\\multicolumn{{4}}{{c}}{{\\textbf{{Photometric Bias by Band ($\\text{{Bias}}_{{\\text{{norm}}}}$)}}}} \\\\
\\hline
Blue Band Bias & {avg_blue_bias:+.3f}\\% & 0.0\\% & Energy conservation in Gaia BP photometer \\\\
Green Band Bias & {avg_green_bias:+.3f}\\% & 0.0\\% & Energy conservation in central optical band \\\\
Red Band Bias & {avg_red_bias:+.3f}\\% & 0.0\\% & Energy conservation in Gaia RP photometer \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    print("\n" + "="*60)
    print(" GENERATED EVALUATION TABLE (LATEX)")
    print("="*60)
    print(latex_table)
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the LaTeX table.
        latex_path = os.path.join(output_dir, "evaluation_metrics.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"[✓] LaTeX file saved to: {latex_path}")

        # Render the metrics table as a PNG image.
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("tight")
        ax.axis("off")
        
        table_data = [
            ["Peak-Normalized MAE", f"{avg_pn_mae:.3f}%", "0.0%", "Global precision independent of magnitude"],
            ["Normalized Std. Dev. (NSTD)", f"{avg_std:.3f}%", "0.0%", "Statistical dispersion of flux residuals"],
            ["Reduced Chi-Square", f"{avg_chi_sq:.3f}", "~1.0", "Statistical validity relative to noise"],
            ["Normalized Bias (MBE)", f"{avg_bias:+.3f}%", "0.0%", "Integrated photometric flux conservation"],
            ["Cosine Similarity", f"{avg_cos_sim:.5f}", "1.00000", "Macroscopic continuum morphological fidelity"],
            ["Blue Band MAE (< 500 nm)", f"{avg_blue_mae:.3f}%", "Minimum", "Regional response in Gaia BP photometer"],
            ["Green Band MAE (500 - 700 nm)", f"{avg_green_mae:.3f}%", "Minimum", "Regional response in central optical band"],
            ["Red Band MAE (> 700 nm)", f"{avg_red_mae:.3f}%", "Minimum", "Regional response in Gaia RP photometer"],
            ["Blue Band Bias (< 500 nm)", f"{avg_blue_bias:+.3f}%", "0.0%", "Photometric energy conservation in Gaia BP"],
            ["Green Band Bias (500 - 700 nm)", f"{avg_green_bias:+.3f}%", "0.0%", "Photometric energy conservation in central band"],
            ["Red Band Bias (> 700 nm)", f"{avg_red_bias:+.3f}%", "0.0%", "Photometric energy conservation in Gaia RP"]
        ]

        columns = ["Metric", "Mean", "Objective", "Utility"]
        col_widths = [0.22, 0.10, 0.10, 0.56]
        
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            colWidths=col_widths,
            cellLoc="left",
            loc="center"
        )

        for i in range(len(table_data) + 1):
            table[(i, 1)].set_text_props(ha="center")
            table[(i, 2)].set_text_props(ha="center")

        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.0, 2)

        plt.title(f"Quantitative Evaluation of the {model_name} Model (Evaluation sample: {num_stars} stars)",
                  fontsize=14, fontweight="bold", pad=20)
        
        png_path = os.path.join(output_dir, "evaluation_metrics.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"[✓] PNG Image saved to: {png_path}")
        print("="*60 + "\n")
