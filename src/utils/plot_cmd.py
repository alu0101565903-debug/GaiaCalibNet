import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Silence user warnings for a cleaner execution output
warnings.filterwarnings('ignore')

def plot_color_magnitude_diagram(input_csv: str, output_image: str):
    """
    Generates a Color-Magnitude Diagram (CMD) from a crossmatched CSV file 
    containing CALSPEC and Gaia DR3 data.
    
    It filters out invalid parallaxes, calculates absolute magnitudes, 
    plots the data, and annotates specific benchmark white dwarfs.

    Parameters:
    -----------
    input_csv : str
        Path to the input CSV file containing the crossmatched data.
    output_image : str
        Path where the resulting plot image will be saved.
    """
    print("=" * 60)
    print("[3/3] GENERATING COLOR-MAGNITUDE DIAGRAM WITH LABELS")
    print("=" * 60)
    
    if not os.path.exists(input_csv):
        print(f"[!] Error: File {input_csv} does not exist.")
        return

    # Read the CSV file, treating '--' and empty strings as NaN
    df = pd.read_csv(input_csv, na_values=['--', ''])
    
    # 1. Numeric coercion: Force scientific columns to be numeric types.
    # The 'coerce' parameter turns invalid parsing into NaN automatically.
    df['parallax'] = pd.to_numeric(df['parallax'], errors='coerce')
    df['phot_g_mean_mag'] = pd.to_numeric(df['phot_g_mean_mag'], errors='coerce')
    df['bp_rp'] = pd.to_numeric(df['bp_rp'], errors='coerce')
    
    # 2. Duplicate filter: Keep only the first occurrence if multiple 
    # CALSPEC stars resolve to the same Gaia source ID (unresolved binaries).
    df_unique = df.drop_duplicates(subset=['gaia_source_id'], keep='first')
    
    # 3. Quality filter: Drop rows with missing crucial data and 
    # keep only stars with positive parallaxes.
    df_clean = df_unique.dropna(subset=['parallax', 'phot_g_mean_mag', 'bp_rp'])
    df_clean = df_clean[df_clean['parallax'] > 0].copy()
    
    print(f"-> Processing {len(df_clean)} unique stars with valid parallax...")

    # 4. Calculate distance in parsecs and Absolute Magnitude (M_G)
    df_clean['distance_pc'] = 1000.0 / df_clean['parallax']
    df_clean['abs_mag_g'] = df_clean['phot_g_mean_mag'] - 5 * np.log10(df_clean['distance_pc']) + 5

    # 5. Plot the diagram
    # Make the figure slightly larger (12x9) to accommodate text boxes safely
    plt.figure(figsize=(12, 9)) 
    
    # In astronomy, the Y-axis is always inverted (lower magnitudes = brighter)
    plt.gca().invert_yaxis()

    scatter = plt.scatter(df_clean['bp_rp'], df_clean['abs_mag_g'], 
                          c=df_clean['bp_rp'], cmap='RdYlBu_r', 
                          s=60, edgecolors='black', alpha=0.8)

    # ==========================================
    # 6. BENCHMARK STARS ANNOTATION
    # ==========================================
    # List is kept lower-case for uniformity during comparison
    stars_to_label = ['wd0501+527', 'gd71', 'gd153', 'hz43', 'vb8']
    
    for star in stars_to_label:
        # Case-insensitive matching to avoid issues with catalog name conventions
        match = df_clean[df_clean['calspec_name'].str.lower() == star]
        
        if not match.empty:
            x_pos = match['bp_rp'].values[0]
            y_pos = match['abs_mag_g'].values[0]
            
            # Change the technical name to the common name for G191-B2B, uppercase the rest
            label_name = 'G191-B2B' if star == 'wd0501+527' else star.upper()
            
            # Add annotation with a pointing arrow
            plt.annotate(label_name, 
                         (x_pos, y_pos), 
                         textcoords="offset points", 
                         xytext=(15, 10), # Text offset (X, Y)
                         ha='left', 
                         fontsize=10,
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                         arrowprops=dict(arrowstyle="->", color='black', alpha=0.7, lw=1.5))

    # Visual configuration
    plt.title('Color-Magnitude Diagram (CALSPEC x Gaia DR3)', fontsize=16, fontweight='bold')
    plt.xlabel('Color BP - RP (Bluer -> Redder)', fontsize=12)
    plt.ylabel('Absolute Magnitude M_G (Luminosity)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('BP-RP Color Index')

    # 7. Save and show the plot
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    
    print(f"-> Success! Plot with labels saved at: {output_image}")
    plt.show()

if __name__ == "__main__":
    input_file = "data/calspec_gaia_dr3_crossmatch.csv"
    output_image = "data/cmd_calspec.png"
    plot_color_magnitude_diagram(input_file, output_image)