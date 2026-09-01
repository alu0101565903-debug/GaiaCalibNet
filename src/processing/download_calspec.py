+"""
Module: download_calspec.py
Report Section: 4.1 CALSPEC Data Acquisition

This script downloads absolute flux-calibrated 1D spectral FITS files from the 
STScI CALSPEC repository for targets successfully cross-matched with Gaia DR3. 
It uses multithreaded requests for concurrent data retrieval and includes 
file-integrity guards to prevent zero-byte corrupt downloads.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import requests


def download_single_file(fits_file: str, base_url: str, output_dir: Path) -> str:
    """
    Download a single CALSPEC FITS spectrum file from the STScI remote archive.

    This helper is executed within thread pool workers. It checks if the asset 
    already exists locally, streams the remote file to disk with network timeout 
    guards, and cleans up incomplete files if an HTTP or transfer error occurs.

    Parameters
    ----------
    fits_file : str
        Filename of the CALSPEC FITS spectrum to retrieve.
    base_url : str
        Base URL endpoint of the remote CALSPEC archive repository.
    output_dir : Path
        Local directory path where the downloaded file will be saved.

    Returns
    -------
    str
        Status string indicating outcome: 'exists', 'downloaded', or 'failed'.
    """
    local_path = output_dir / fits_file

    # Verify existing non-empty file to avoid redundant network requests
    if local_path.exists() and local_path.stat().st_size > 0:
        return "exists"

    url = f"{base_url}{fits_file}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        # Stream response in chunks to optimize RAM usage for large FITS files
        response = requests.get(url, headers=headers, stream=True, timeout=20)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return "downloaded"
    except Exception as e:
        print(f"  [!] Failed downloading asset {fits_file}: {e}")
        # Remove partial or corrupt file remnants
        if local_path.exists():
            try:
                local_path.unlink()
            except OSError:
                pass
        return "failed"


def download_missing_fits(
    csv_path: Union[str, Path],
    output_dir_str: Union[str, Path],
    max_workers: int = 10,
) -> None:
    """
    Read the crossmatch catalog and download missing CALSPEC FITS files in parallel.

    Extracts target file names, filters out duplicates, and downloads missing 
    assets using a thread pool. Pre-existing non-empty files are preserved.

    Parameters
    ----------
    csv_path : str or Path
        File path to the processed crossmatch catalog CSV.
    output_dir_str : str or Path
        Directory path where CALSPEC FITS files will be stored.
    max_workers : int, optional
        Maximum number of parallel thread workers. Default is 10.

    Returns
    -------
    None

    Notes
    -----
    Target base repository:
    https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/
    """
    print("=" * 60)
    print("[4.1] DOWNLOADING TARGET SPECTRAL FITS FILES FROM STScI")
    print("=" * 60)

    csv_file = Path(csv_path)
    output_dir = Path(output_dir_str)

    if not csv_file.exists():
        print(f"[!] Error: Crossmatch catalog not found at: {csv_path}")
        print('    Please run "crossmatch.py" before executing this download task.')
        return

    df = pd.read_csv(csv_file)
    if "fits_file" not in df.columns:
        print("[!] Error: Column 'fits_file' missing from the crossmatch catalog.")
        return

    # Extract unique valid FITS entries
    required_files = [str(f).strip() for f in df["fits_file"].dropna().unique() if str(f).strip()]
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"

    print(f"-> Detected {len(required_files)} target stars required by the catalog.")
    print("-> Initiating concurrent parallel download...\n")

    stats: Dict[str, int] = {"downloaded": 0, "exists": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_file, fits_name, base_url, output_dir): fits_name
            for fits_name in required_files
        }

        for future in as_completed(futures):
            fits_name = futures[future]
            try:
                status = future.result()
                stats[status] = stats.get(status, 0) + 1
                if status == "downloaded":
                    print(f"  [✓] Successfully downloaded: {fits_name}")
            except Exception as e:
                print(f"  [!] Worker thread error for {fits_name}: {e}")
                stats["failed"] += 1

    total_valid_fits = len(list(output_dir.glob("*.fits"))) + len(list(output_dir.glob("*.fit")))

    print("\n" + "=" * 40)
    print("        DOWNLOAD EXECUTION SUMMARY")
    print("=" * 40)
    print(f"Pre-existing files verified locally  : {stats['exists']}")
    print(f"Successfully downloaded files        : {stats['downloaded']}")
    print(f"Failed downloads                     : {stats['failed']}")
    print(f"Total valid spectra available in pool: {total_valid_fits}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    crossmatch_csv = "data/processed/crossmatch_catalog.csv"
    spectra_output_dir = "data/raw/calspec_fits"

    download_missing_fits(crossmatch_csv, spectra_output_dir)