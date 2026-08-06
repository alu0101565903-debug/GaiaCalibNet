"""
Module: build_calspec_reference.py
Report Section:

CALSPEC Reference Catalog Extractor.

Automates downloading and extracting the STScI CALSPEC flux standard calibration 
reference table. Scrapes the official STScI portal, parses the embedded astronomical 
catalogs, isolates essential features (Object Name and Spectral Type), cleans missing/invalid 
records, and exports the sanitized dataset as a raw CSV artifact.

Output Artifact:
- CSV file containing "Name" and "Sp. T." columns at "data/raw/calspec_reference.csv".
"""

from __future__ import annotations

import io
import os
import pandas as pd
import requests

# Official STScI CALSPEC database endpoint
CALSPEC_URL = (
    "https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec"
)

# Destination directory and filename for the cleaned reference table
OUTPUT_PATH = "data/raw/calspec_reference.csv"


def build_reference_table(
    url: str = CALSPEC_URL, 
    output_path: str = OUTPUT_PATH
) -> None:
    """
    Scrape, clean, and export the minimal CALSPEC reference table.

    Fetches the primary HTML document from STScI, parses all embedded tables to 
    locate the target CALSPEC database, isolates required columns ("Name" and "Sp. T."), 
    filters out missing or placeholder records, deduplicates entries, and writes the 
    result to disk.

    Parameters
    - url (str, default=CALSPEC_URL): Target web URL for the official STScI CALSPEC catalog page.
    - output_path (str, default=OUTPUT_PATH): Local destination file path where the sanitized CSV artifact will be saved.

    Raises
    - RuntimeError: If no table containing both required columns ("Name", "Sp. T.") is found.
    - requests.RequestException: If the HTTP GET request fails or times out.
    """
    print(f"[>] Downloading CALSPEC reference page from STScI...")

    # 1. Fetch raw HTML content with a 30-second timeout safeguard
    response = requests.get(CALSPEC_URL, timeout=30)
    response.raise_for_status()

    # 2. Parse embedded HTML tables using pandas read_html
    print("[>] Searching for CALSPEC tables in parsed HTML...")
    tables = pd.read_html(io.StringIO(response.text))

    target_table: pd.DataFrame | None = None

    # 3. Locate the table containing the required spectral columns
    for table in tables:
        cols = {str(c).strip() for c in table.columns}

        if "Name" in cols and "Sp. T." in cols:
            target = table
            break

    if target is None:
        raise RuntimeError(
            "Could not find a table containing 'Name' and 'Sp. T.' columns."
        )

    # 4. Extract required features
    reference = target[["Name", "Sp. T."]].copy()

    # 5. Data Hygiene: Normalize strings and strip trailing whitespace
    reference["Name"] = reference["Name"].astype(str).str.strip()
    reference["Sp. T."] = reference["Sp. T."].astype(str).str.strip()

    # 6. Filter empty strings and common missing-data placeholders ("...")
    reference = reference[
        (reference["Name"] != "")
        & (reference["Sp. T."] != "")
        & (reference["Sp. T."] != "...")
    ]

    # 7. Deduplicate records based on target astronomical name
    reference = reference.drop_duplicates(subset="Name")

    # 8. Ensure output directory exists before saving
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 9. Export sanitized reference catalog
    reference.to_csv(OUTPUT_PATH, index=False)

    print(f"[✓] CALSPEC reference table successfully exported to: {output_path}")
    print(f"[✓] Total valid stellar records: {len(reference)}\n")
    print(reference.head())


if __name__ == "__main__":
    build_reference_table()