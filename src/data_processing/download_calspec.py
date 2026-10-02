"""Download flux-calibrated CALSPEC spectra for cross-matched targets.

Files are retrieved from the STScI archive using multiple worker threads and
validated to avoid retaining incomplete downloads.

Notes
-----
Report context: Section 4.1, CALSPEC Data Acquisition.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

def download_single_file(fits_file: str, base_url: str, output_dir: Path) -> str:
    """
    Download a single CALSPEC FITS file from the remote archive.

    This helper is designed to run in a separate thread. It checks whether the
    file already exists locally, then streams the remote file to disk with a
    timeout and a non-zero content guard.

    Parameters
    ----------
    fits_file : str
        Name of the CALSPEC FITS file to retrieve.
    base_url : str
        Remote base URL for the CALSPEC archive.
    output_dir : pathlib.Path
        Local directory path where the file will be saved.

    Returns
    -------
    status : str
        Status result string: 'exists', 'downloaded', or 'failed'.
    """
    local_path = output_dir / fits_file

    # Build the local destination path and check whether the file has already
    # been successfully downloaded. This avoids repeating downloads on reruns.
    if local_path.exists() and local_path.stat().st_size > 0:
        return "exists"

    url = f"{base_url}{fits_file}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        # Stream the remote file in chunks to avoid loading the entire response
        # into memory, which is important for large FITS files.
        response = requests.get(url, headers=headers, stream=True, timeout=20)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return "downloaded"
    except Exception as e:
        print(f"  [!] Failed downloading asset {fits_file}: {e}")
        # Remove any partially written file to prevent future false positives.
        if local_path.exists():
            local_path.unlink()
        return "failed"

def download_missing_fits(csv_path: str, output_dir_str: str) -> None:
    """
    Read the crossmatch catalog and download any missing CALSPEC FITS files.

    The function verifies that the crossmatch catalog exists, extracts the list
    of target FITS file names, and downloads missing files in parallel using a
    thread pool. Existing non-empty files are preserved.

    Parameters
    ----------
    csv_path : str
        Path to the processed crossmatch catalog CSV.
    output_dir_str : str
        Directory where CALSPEC FITS files will be stored.

    Returns
    -------
    None
    """
    print("=" * 60)
    print("[4.1] DOWNLOADING TARGET SPECTRAL FITS FILES FROM STScI")
    print("=" * 60)

    csv_file = Path(csv_path)
    output_dir = Path(output_dir_str)

    if not csv_file.exists():
        print(f"[!] Error: Crossmatch catalog not found at {csv_path}")
        print('Please run "crossmatch_wd.py" before executing this download task.')
        return

    df = pd.read_csv(csv_file)
    if "fits_file" not in df.columns:
        print("[!] Error: Column 'fits_file' missing from the crossmatch catalog.")
        return

    # Remove duplicate file names and ignore missing FITS entries. This ensures
    # each target file is downloaded at most once.
    required_files = df["fits_file"].dropna().unique()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"

    print(f"-> Detected {len(required_files)} target stars required by the catalog.")
    print("-> Initiating concurrent parallel download...\n")

    stats = {"downloaded": 0, "exists": 0, "failed": 0}

    # Use a moderate number of worker threads to balance download speed and
    # polite access to the STScI archive.
    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_file, f.strip(), base_url, output_dir): f.strip()
            for f in required_files
        }

        # Collect results as each download task completes. This uses the
        # completed futures in real time, so the progress output appears live.
        for future in as_completed(futures):
            fits_name = futures[future]
            try:
                status = future.result()
                stats[status] += 1
                if status == "downloaded":
                    print(f"  [✓] Successfully downloaded: {fits_name}")
            except Exception as e:
                # If a worker thread raises outside the download logic, count it as failed.
                print(f"  [!] Thread crashed for {fits_name}: {e}")
                stats["failed"] += 1

    print("\n" + "=" * 40)
    print("       DOWNLOAD EXECUTION SUMMARY")
    print("=" * 40)
    print(f"Pre-existing files verified locally  : {stats['exists']}")
    print(f"Successfully downloaded files        : {stats['downloaded']}")
    print(f"Failed downloads                     : {stats['failed']}")
    print(f"Total valid spectra available in pool: {len(list(output_dir.glob('*.fits')))}")
    print("=" * 60 + "\n")
    
if __name__ == "__main__":
    crossmatch_csv = "data/processed/training_catalog.csv"
    spectra_output_dir = "data/raw/calspec_fits"
    
    download_missing_fits(crossmatch_csv, spectra_output_dir)