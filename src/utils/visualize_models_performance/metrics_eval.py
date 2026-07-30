"""
Module: metrics_eval.py

Quantitative evaluation module for the PINN spectral recalibration pipeline.

This module provides tools to load exported model predictions and evaluate the 
model's performance using standard astrophysical metrics. It computes Peak-Normalized 
Mean Absolute Error (MAE), Reduced Chi-Square, Normalized Bias (MBE), Cosine Similarity, 
and relative errors grouped by specific wavelength bands. The results are aggregated 
and automatically exported as both a LaTeX table and a PNG image for direct inclusion 
in academic reports.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.metrics.alternative_metrics import (
    calculate_peak_normalized_mae,
    calculate_reduced_chi_square,
    calculate_normalized_bias_percentage,
    calculate_cosine_similarity,
    calculate_band_errors
)

def evaluate_test_set_from_file(input_filepath: str, output_dir: str, num_knots: int = 20):
    """
    Reads predictions from disk and evaluates the PINN model's performance using 
    astrophysical metrics. Exports a LaTeX code snippet and a PNG image of the results.

    This function iterates through all stellar samples stored in the input `.npz` archive,
    calculates multiple quantitative metrics (Peak-Normalized MAE, Reduced Chi-Square, 
    Normalized Bias, Cosine Similarity, and Band-specific Errors) for each sample, 
    and computes the global averages. The final aggregated metrics are formatted 
    into a professional LaTeX table and rendered as a high-resolution PNG image.

    Args:
    - input_filepath (str): The full path to the `.npz` file containing the 
        exported inference data (original spectra, references, predictions, and errors).
    - output_dir (str): The destination directory where the generated `.tex` 
        and `.png` files will be saved. The directory is created if it does not exist.
    - num_knots (int, optional): The number of free parameters (or control knots) 
        used by the model, required to properly calculate the degrees of freedom 
        for the Reduced Chi-Square metric. Defaults to 20.

    Returns:
    - None: This function does not return any Python objects. It writes the evaluation 
        results directly to the standard output and the filesystem.

    Raises:
    - FileNotFoundError: If the `.npz` file specified by `input_filepath` does not exist.
    """
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"File not found: {input_filepath}")
        
    # 1. Load data from disk
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
    
    # 2. Initialize accumulators
    pn_mae_list, chi_sq_list, bias_list, cos_sim_list = [], [], [], []
    band_metrics_accum = {"blue": 0.0, "green": 0.0, "red": 0.0}
    
    # 3. Calculate metrics star by star
    for i in range(num_stars):
        pn_mae_list.append(calculate_peak_normalized_mae(y_true[i], y_pred[i]))
        chi_sq_list.append(calculate_reduced_chi_square(y_true[i], y_pred[i], y_err[i], n_params=num_knots))
        bias_list.append(calculate_normalized_bias_percentage(y_true[i], y_pred[i]))
        cos_sim_list.append(calculate_cosine_similarity(y_true[i], y_pred[i]))
        
        bands = calculate_band_errors(wave, y_true[i], y_pred[i])
        band_metrics_accum["blue"] += bands["blue_rel_error_percent"]
        band_metrics_accum["green"] += bands["green_rel_error_percent"]
        band_metrics_accum["red"] += bands["red_rel_error_percent"]

    # 4. Final Averages
    avg_pn_mae = np.mean(pn_mae_list)
    avg_chi_sq = np.mean(chi_sq_list)
    avg_bias = np.mean(bias_list)
    avg_cos_sim = np.mean(cos_sim_list)
    avg_blue = band_metrics_accum["blue"] / num_stars
    avg_green = band_metrics_accum["green"] / num_stars
    avg_red = band_metrics_accum["red"] / num_stars

    # --- 5. LaTeX Code Generation ---
    latex_table = f"""
\\begin{{table}}[htpb]
\\centering
\\caption{{Quantitative Evaluation Results of the PINN Model (Blind Test: {num_stars} stars)}}
\\label{{tab:pinn_evaluation}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} & \\textbf{{Unit}} \\\\
\\hline
Peak-Normalized MAE & {avg_pn_mae:.3f} & \\% \\\\
Reduced $\\chi^2_\\nu$ & {avg_chi_sq:.3f} & - \\\\
Normalized Bias (MBE) & {avg_bias:.3f} & \\% \\\\
Cosine Similarity & {avg_cos_sim:.5f} & - \\\\
\\hline
\\multicolumn{{3}}{{c}}{{\\textbf{{Relative Error by Band}}}} \\\\
\\hline
Blue Band (< 500 nm) & {avg_blue:.3f} & \\% \\\\
Green Band (500 - 700 nm) & {avg_green:.3f} & \\% \\\\
Red Band (> 700 nm) & {avg_red:.3f} & \\% \\\\
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
        
        # Save LaTeX string to file
        latex_path = os.path.join(output_dir, "evaluation_metrics.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"[✓] LaTeX file saved to: {latex_path}")

        # --- 6. PNG Image Generation with Matplotlib ---
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("tight")
        ax.axis("off")
        
        table_data = [
            ["Peak-Normalized MAE", f"{avg_pn_mae:.3f}", "%"],
            ["Reduced Chi-Square", f"{avg_chi_sq:.3f}", "-"],
            ["Normalized Bias (MBE)", f"{avg_bias:.3f}", "%"],
            ["Cosine Similarity", f"{avg_cos_sim:.5f}", "-"],
            ["Blue Band (< 500 nm)", f"{avg_blue:.3f}", "%"],
            ["Green Band (500 - 700 nm)", f"{avg_green:.3f}", "%"],
            ["Red Band (> 700 nm)", f"{avg_red:.3f}", "%"]
        ]
        columns = ["Metric", "Value", "Unit"]
        
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc="center",
            loc="center"
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5) 

        plt.title(f"Quantitative Evaluation of the PINN Model (Blind Test: {num_stars} stars)", 
                  fontsize=14, fontweight="bold", pad=20)
        
        png_path = os.path.join(output_dir, "evaluation_metrics.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"[✓] PNG Image saved to: {png_path}")
        print("="*60 + "\n")