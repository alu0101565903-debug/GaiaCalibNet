"""
Module: catalogs_by_color.py
Report Section: 

CALSPEC Spectral Type Categorization & Color Catalog Generator.

Parses spectral classification strings for CALSPEC flux standard stars and partitions 
crossmatched catalogs into three photometric regime subsets:
  - Blue  : Early-type stars (O, B, A), White Dwarfs (DA, DB, DO), and Hot Subdwarfs (sdO, sdB).
  - Solar : Solar-type G and F stars, plus Solar-like Subdwarfs (sdF).
  - Red   : Late-type cool stars (K, M) and substellar objects (L, T, Y).

Classification Rules (Morgan-Keenan Scheme & Exceptions):
1. Binary/Composite Types: Truncated at first indicator ("+" or "/") to prioritize primary target.
2. Hot Compact Objects: "SDO", "SDB", "DA", "DB", "DO" -> Forced "Blue".
3. Solar Subdwarfs: "SDF" -> Forced "Solar".
4. Primary Letters:
     - "O", "B", "A"            -> "Blue"
     - "F", "G"                 -> "Solar"
     - "K", "M", "L", "T", "Y"  -> "Red"
5. Missing/Unrecognized values -> "Unknown"
"""

from __future__ import annotations

import os
import pandas as pd


def load_spectral_lookup(calspec_table_path: str) -> dict[str, str]:
    """"
    Build a case-insensitive lookup dictionary mapping object names to spectral types.

    Parameters
    - calspec_table_path (str)
        Path to the reference CSV file containing CALSPEC metadata. 
        Must include columns "Name" and "Sp. T.".

    Returns
    - Dict[str, str]: Dictionary formatted as `{normalized_name: spectral_type_string}`.

    Raises
    - FileNotFoundError: If `calspec_table_path` does not exist.
    - ValueError: If mandatory columns ("Name", "Sp. T.") are missing from the input file.
    """
    if not os.path.exists(calspec_table_path):
        raise FileNotFoundError(f"CALSPEC table not found at: {calspec_table_path}")
    
    df = pd.read_csv(calspec_table_path)

    required = {"Name", "Sp. T."}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in CALSPEC table: {missing}"
        )

    lookup: dict[str, str] = {}

    for _, row in df.iterrows():
        name = str(row["Name"]).strip().lower()
        sp = str(row["Sp. T."]).strip()

        # Omit invalid entries and STScI placeholder strings ("...")
        if name and sp and sp != "...":
            lookup[name] = sp

    return lookup


def spectral_type_to_color(sp_type: str) -> str:
    """
    Map a Morgan-Keenan or degenerate spectral type string to a color category.

    Parameters
    - sp_type (str): Raw spectral classification string (e.g., "G2V", "DA3", "B5III+DV", "sdF8").

    Returns
    - str: One of the color labels: "Blue", "Solar", "Red", or "Unknown".
    """
    if pd.isna(sp_type):
        return "Unknown"

    s = str(sp_type).upper().strip()

    if not s or s == "...":
        return "Unknown"

    # Remove internal whitespaces (e.g., "G 2 V" -> "G2V")
    s = s.replace(" ", "")

    # Clean binary/composite spectral types by isolating the primary component
    s = s.split("+")[0]
    s = s.split("/")[0]

    # Explicit handling for hot compact stars & white dwarfs   
    if s.startswith(("SDO", "SDB", "DA", "DB", "DO")):
        return "Blue"

    # Explicit handling for solar-like metal-poor subdwarfs
    if s.startswith("SDF"):
        return "Solar"

    # Standard Morgan-Keenan first-character classification
    first_char = s[0]

    if first_char in ("O", "B", "A"):
        return "Blue"

    if first_char in ("F", "G"):
        return "Solar"

    if first_char in ("K", "M", "L", "T", "Y"):
        return "Red"

    return "Unknown"


def get_spectral_type(
    calspec_name: str,
    lookup: dict[str, str],
) -> str:
    """
    Retrieve the spectral type string for a given CALSPEC object name.

    Parameters
    - calspec_name (str): Target star identifier in CALSPEC (case-insensitive).
    - lookup (Dict[str, str]): Lookup dictionary loaded via `load_spectral_lookup`.

    Returns
    - str: Spectral type string if present in lookup, otherwise "Unknown".
    """
    key = str(calspec_name).strip().lower()
    return lookup.get(key, "Unknown")


def group_catalog_by_color(
    df: pd.DataFrame,
    spectral_lookup: dict[str, str],
    name_column: str = "calspec_name",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Annotate a crossmatch catalog DataFrame with spectral types and color classes.

    Parameters
    - df (pd.DataFrame): Input catalog containing object identifiers.
    - spectral_lookup (Dict[str, str]): Dictionary mapping object names to spectral types.
    - name_column (str, default="calspec_name"): DataFrame column name containing the object identifiers.

    Returns
    - Tuple[pd.DataFrame, List[str]]
        - Annotated copy of input DataFrame with added "spectral_type" 
          and "color_category" columns.
        - List of primary valid color categories: ["Solar", "Blue", "Red"].
    """
    out = df.copy()

    # Map name -> spectral_type string
    out["spectral_type"] = out[name_column].apply(
        lambda n: get_spectral_type(n, spectral_lookup)
    )

    # Map spectral_type -> color label
    out["color_category"] = out["spectral_type"].apply(
        spectral_type_to_color
    )

    labels = ["Solar", "Blue", "Red"]
    return out, labels


def save_color_catalogs(
    input_csv_path: str,
    calspec_table_path: str,
    output_dir: str,
    name_column: str = "calspec_name",
) -> None:
    """
    Load a crossmatched catalog, partition it by color regime, and write CSV outputs.

    Parameters
    - input_csv_path (str): Filepath to the primary crossmatched catalog CSV.
    - calspec_table_path (str): Filepath to the CALSPEC reference catalog CSV.
    - output_dir (str): Destination directory where sub-catalogs will be created.
    - name_column (str, default="calspec_name"): Column name in `input_csv_path` used for target matching.

    Raises
    - FileNotFoundError: If `input_csv_path` or `calspec_table_path` cannot be located on disk.
    """

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(input_csv_path)

    if not os.path.exists(calspec_table_path):
        raise FileNotFoundError(calspec_table_path)

    print(f"[>] Loading catalogs and reference tables...")
    df = pd.read_csv(input_csv_path)
    spectral_lookup = load_spectral_lookup(calspec_table_path)

    # Classify catalog entries
    print(f"[>] Classifying targets into color regimes...")
    df_classified, labels = group_catalog_by_color(
        df,
        spectral_lookup=spectral_lookup,
        name_column=name_column,
    )

    # Create destination directory structure
    os.makedirs(output_dir, exist_ok=True)
    print(f"[✓] Output directory ready: {output_dir}")

    # Export catalog subset per color category
    for label in labels:
        sub_df = df_classified[
            df_classified["color_category"] == label
        ].drop(columns=["color_category"])

        output_file = os.path.join(
            output_dir,
            f"{label.lower()}_catalog.csv"
        )

        sub_df.to_csv(output_file, index=False)
        print(f"    • {label} Catalog: {len(sub_df)} targets -> {output_file}")


if __name__ == "__main__": 
    # Default pipeline paths
    input_catalog_path = "data/processed/crossmatch_catalog.csv" 

    # CSV containing at least columns: Name, Sp. T. 
    calspec_table_path = "data/raw/calspec_reference.csv" 
    output_directory = "data/processed/catalogs_by_color"

    save_color_catalogs( 
        input_csv_path=input_catalog_path, 
        calspec_table_path=calspec_table_path, 
        output_dir=output_directory, 
        name_column="calspec_name"
    )