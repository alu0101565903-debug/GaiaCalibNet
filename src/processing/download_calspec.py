"""
Module: download_calspec.py
Report Section: 4.1 CALSPEC Data Acquisition

This script downloads the absolute flux-calibrated 1D spectral FITS files 
from the STScI CALSPEC repository for the targets successfully crossmatched 
with the Gaia DR3 framework. It retrieves files sequentially and avoids 
redundant network calls by checking local storage prior to retrieval.
"""

from pathlib import Path
import pandas as pd
from urllib.request import urlretrieve

def download_missing_fits(csv_path: str, output_dir_str: str):
    """
    Reads the processed crossmatch catalog and retrieves missing spectral 
    FITS files from the official STScI server sequentially.

    Parameters:
    - csv_path (str): Path to the processed crossmatch CSV catalog containing the target FITS filenames.
    - output_dir_str (str): Directory path where the downloaded .fits files will be stored.
    """
    print("=" * 60)
    print("[4.1] DOWNLOADING TARGET SPECTRAL FITS FILES FROM STScI")
    print("=" * 60)

    # 1. Path definitions using modern pathlib
    csv_file = Path(csv_path)
    output_dir = Path(output_dir_str)

    # 2. Pre-execution checks
    if not csv_file.exists():
        print(f"[!] Error: Crossmatch catalog not found at {csv_path}")
        print('Please run "crossmatch_wd.py" before executing this download task.')
        return

    # Load the processed data
    df = pd.read_csv(csv_file)
    if "fits_file" not in df.columns:
        print("[!] Error: Column 'fits_file' missing from the crossmatch catalog.")
        return
    
    # Isolate unique FITS files to avoid duplicate downloads
    required_files = df["fits_file"].dropna().unique()
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Official STScI verified data repository link
    base_url = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
    
    print(f"-> Detected {len(required_files)} target stars required by the catalog.")
    print("-> Initiating sequential download...\n")

    # 3. Execution metrics tracker
    stats = {"downloaded": 0, "exists": 0, "failed": 0}
    
    # 4. Sequential download loop
    for fits_file in required_files:
        fits_file = fits_file.strip()
        local_path = output_dir / fits_file
        
        # Skip download if the file is already stored locally
        if local_path.exists():
            stats["exists"] += 1
            continue
            
        # Structure the explicit URL targeting the remote FITS asset
        url = f"{base_url}{fits_file}"
        
        try:
            print(f"  -> Downloading: {fits_file}...")
            urlretrieve(url, local_path)
            stats["downloaded"] += 1
        except Exception as e:
            print(f"  [!] Failed downloading asset {fits_file}: {e}")
            stats["failed"] += 1

    # 5. Final summary reporting
    print("\n" + "=" * 40)
    print("       DOWNLOAD EXECUTION SUMMARY")
    print("=" * 40)
    print(f"Pre-existing files verified locally  : {stats['exists']}")
    print(f"Successfully downloaded files        : {stats['downloaded']}")
    print(f"Failed downloads                     : {stats['failed']}")
    print(f"Total valid spectra available in pool: {len(list(output_dir.glob('*.fits')))}")
    print("=" * 60 + "\n")
    
if __name__ == "__main__":
    # Path routing aligned with the project's data management architecture
    crossmatch_csv = "data/processed/crossmatch_catalog.csv"
    spectra_output_dir = "data/raw/calspec_fits"
    
    download_missing_fits(crossmatch_csv, spectra_output_dir)