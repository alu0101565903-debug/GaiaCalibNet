import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Silence user warnings for a cleaner execution output
warnings.filterwarnings("ignore")

def plot_color_magnitude_diagram(input_csv: str, output_image: str):
    """
    Generate a color-magnitude diagram from cross-matched CALSPEC and Gaia DR3 data.
    
    It filters out invalid parallaxes, calculates absolute magnitudes, 
    plots the data, and annotates specific benchmark white dwarfs.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file containing the crossmatched data.
    output_image : str
        Path where the resulting plot image will be saved.

    Returns
    -------
    None
    """
    print("=" * 60)
    print("[3/3] GENERATING COLOR-MAGNITUDE DIAGRAM WITH LABELS")
    print("=" * 60)
    
    if not os.path.exists(input_csv):
        print(f"[!] Error: File {input_csv} does not exist.")
        return

    # Read the CSV file, treating "--" and empty strings as NaN.
    df = pd.read_csv(input_csv, na_values=["--", ""])
    
    # 1. Convert scientific columns to numeric values; invalid values become NaN.
    df["parallax"] = pd.to_numeric(df["parallax"], errors="coerce")
    df["phot_g_mean_mag"] = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce")
    df["bp_rp"] = pd.to_numeric(df["bp_rp"], errors="coerce")
    
    # 2. Keep the first row when multiple CALSPEC stars resolve to one Gaia source.
    df_unique = df.drop_duplicates(subset=["gaia_source_id"], keep="first")
    
    # 3. Remove rows with missing required data or non-positive parallaxes.
    df_clean = df_unique.dropna(subset=["parallax", "phot_g_mean_mag", "bp_rp"])
    df_clean = df_clean[df_clean["parallax"] > 0].copy()
    
    print(f"-> Processing {len(df_clean)} unique stars with valid parallax...")

    # 4. Calculate distances in parsecs and absolute magnitudes.
    df_clean["distance_pc"] = 1000.0 / df_clean["parallax"]
    df_clean["abs_mag_g"] = df_clean["phot_g_mean_mag"] - 5 * np.log10(df_clean["distance_pc"]) + 5

    # 5. Plot the diagram with room for annotation boxes.
    plt.figure(figsize=(9, 7)) 
    
    scatter = plt.scatter(
        df_clean["bp_rp"],
        df_clean["abs_mag_g"],
        c=df_clean["bp_rp"],
        cmap="RdYlBu_r",
        s=50,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.85,
    )

    # Invert Y-axis after plotting (brighter stars on top)
    plt.gca().invert_yaxis()

    # Visual configuration
    plt.title("CALSPEC $\\times$ Gaia DR3 Color-Magnitude Diagram", fontsize=14, fontweight="bold")
    plt.xlabel("$G_{\\mathrm{BP}} - G_{\\mathrm{RP}}$ [mag]", fontsize=12)
    plt.ylabel("$M_G$ [mag]", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.4)
    cbar = plt.colorbar(scatter)
    cbar.set_label("$G_{\\mathrm{BP}} - G_{\\mathrm{RP}}$", fontsize=11)

    # 6. Save and display the plot.
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    
    print(f"-> Success! Plot with labels saved at: {output_image}")
    plt.show()

if __name__ == "__main__":
    input_file = "data/processed/crossmatch_catalog.csv"
    output_image = "data/processed/cmd_crossmatch.png"     # Change extension to .eps for vector graphics if needed
    plot_color_magnitude_diagram(input_file, output_image)