"""
Module: catalogs_by_color.py

Stellar Catalog Segmentation Module.
This module provides utility functions to partition a master stellar catalog 
into segmented CSV files based on photometric color groups (Blue, Solar, Red),
accounting for Gaia photometric saturation overrides and spectral type conventions.
"""

from pathlib import Path
import pandas as pd
from src.utils.visualize_spectra.catalogs_management import group_catalog_by_color

def split_catalog_by_color(input_csv: str | Path, output_dir: str | Path = ".") -> None:
    """
    Reads the original stellar catalog, classifies stars into color groups via
    group_catalog_by_color, and exports an independent CSV file for each group
    while strictly preserving the original column structure.

    Args:
    - input_csv (str | Path): Path to the input master catalog CSV file.
    - output_dir (str | Path, optional): Target directory where segmented CSV files
        will be saved. Defaults to the current working directory (".").
    """
    input_path = Path(input_csv)
    out_dir = Path(output_dir)

    # Create target directory if it does not exist (including parent folders)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load catalog and apply color/physical classification logic
    df_original = pd.read_csv(input_path)
    df_classified, labels = group_catalog_by_color(df_original)

    # 2. Map classification labels to output filenames within the target directory
    file_mapping = {
        labels[0]: out_dir / "blue_catalog.csv",
        labels[1]: out_dir / "solar_catalog.csv",
        labels[2]: out_dir / "red_catalog.csv",
    }

    # 3. Filter group data, remove temporary classification metadata, and export CSVs
    for label, file_path in file_mapping.items():
        df_group = df_classified[df_classified["Color_Group"] == label].copy()
        
        # Drop temporary classification column to match original catalog formatting exactly
        df_group.drop(columns=["Color_Group"], inplace=True)
        
        df_group.to_csv(file_path, index=False)
        print(f"[✓] File generated: {file_path} ({len(df_group)} stars)")


if __name__ == "__main__":

    input_catalog_path = "data/processed/crossmatch_catalog.csv"
    output_directory = "data/processed/catalogs_by_color"
    
    split_catalog_by_color(input_csv=input_catalog_path, output_dir=output_directory)