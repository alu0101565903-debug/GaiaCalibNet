"""
Module: create_catalog.py
Report Section: 4.1 CALSPEC Data Acquisition & Catalog Generation

This script scrapes the remote STScI CALSPEC server index, organizes files by target,
parses remote FITS headers stream-on-the-fly to extract target names and coordinates,
and leverages a SIMBAD/Sesame fallback parser to resolve missing coordinates using
astronomical name normalization.
"""

import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits

# Ignore Astropy and Pandas runtime warnings for cleaner terminal reporting
warnings.filterwarnings("ignore")

def clean_target_name(name: str, prefix: str) -> str:
    """
    Cleans and standardizes raw target strings to maximize coordinate resolution 
    success rates when querying the SIMBAD/Sesame name resolvers.
    """
    base_name = name.upper() if name else prefix.upper()

    # Regular expressions to enforce standard cross-matching nomenclatures
    base_name = re.sub(r"^WD(\d{4})_(\d{3,4})$", r"WD \1-\2", base_name)
    base_name = re.sub(r"^WDFS(\d{4})_(\d{2})$", r"WDFS \1-\2", base_name)
    base_name = re.sub(r"^NGC(\d+)_(\d+)$", r"NGC \1 \2", base_name)
    base_name = re.sub(r"^BD_?(\d+)D(\d+)$", r"BD+\1 \2", base_name)
    base_name = re.sub(r"^2M(\d{6,})$", r"2MASS J\1", base_name)
    
    # Standardize string dividers
    base_name = base_name.replace("_", " ")
    base_name = re.sub(r"^(\d+)([A-Z]+)$", r"\1 \2", base_name)
    
    return base_name.strip()

def get_coords_sesame(star_name: str):
    """
    Queries the Sesame/SIMBAD gateway using Astropy to resolve SkyCoord coordinates.
    """
    try:
        coord = SkyCoord.from_name(star_name)
        return round(coord.ra.degree, 5), round(coord.dec.degree, 5)
    except Exception:
        return None, None

def is_valid_coord(val) -> bool:
    """
    Evaluates whether an extracted value corresponds to a valid numerical coordinate.
    """
    if val is None or pd.isna(val) or val == "":
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


def process_single_star(prefix: str, files: list, base_url: str) -> dict:
    """
    Worker function to process a single star group. Designed to run in parallel threads.
    """
    # Prioritize observational datasets over theoretical model grids ("mod")
    best_file = sorted(files, key=lambda x: 0 if "mod" not in x.lower() else 1)[0]
    
    try:
        # Stream and pull the remote FITS primary header records
        header = fits.getheader(base_url + best_file)
        raw_name = str(header.get("TARGNAME", prefix)).strip()
        
        # Extract standard spatial coordinates fields
        ra = header.get("RA_TARG", header.get("RA"))
        dec = header.get("DEC_TARG", header.get("DEC"))
        
        # Trigger SIMBAD/Sesame resolution loop if header positions are blank or invalid
        if not is_valid_coord(ra) or not is_valid_coord(dec):
            cleaned_name = clean_target_name(raw_name, prefix)
            ra, dec = get_coords_sesame(cleaned_name)
            
        if ra is not None and dec is not None:
            return {
                "calspec_name": raw_name,
                "RA": float(ra),
                "DEC": float(dec),
                "fits_file": best_file
            }
    except Exception:
        # Silently ignore read/connection errors for this specific file to avoid halting execution
        pass
    return None


def create_catalog(output_path: str):
    """
    Scrapes the remote index, structures targets, handles observational file prioritization,
    resolves coordinates, and dumps the raw consolidated metadata table.
    """
    print("=" * 60)
    print("[4.1] SCRA-PARSING AND GENERATING BASE CALSPEC CATALOG")
    print("=" * 60)
    
    base_url = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        print("-> Querying remote STScI directory index page...")
        req = requests.get(base_url, headers=headers, timeout=15)
        
        # IMPROVEMENT: Ultra-robust FITS link extraction that supports single/double quotes,
        # uppercase/lowercase href values, and relative paths.
        raw_hrefs = re.findall(r'href=["\']([^"\']+)["\']', req.text, re.IGNORECASE)
        fits_files = [os.path.basename(f) for f in raw_hrefs if f.lower().endswith(".fits")]
        
    except Exception as e:
        print(f"[!] Critical Connection Error: Unable to access STScI server: {e}")
        return
    
    # Group file variants matching identical target star roots.
    star_files = {}
    for f in fits_files:
        # FIXED: Add optional version suffix support (e.g. _003 or _v2) in the filename pattern.
        match = re.search(r"^(.+?)_(mod|stis|nic|stiswfcnic|stiswfc|faint)(?:_\w+)?\.fits$", f, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            star_files.setdefault(prefix, []).append(f)
            
    star_data = []
    total_groups = len(star_files)
    
    if total_groups == 0:
        print("[!] Warning: Still found 0 target groups. Here is a sample of found FITS files to diagnose:")
        print(fits_files[:5] if fits_files else "No FITS files extracted from HTML.")
        return

    print(f"-> Found {total_groups} distinct target groups. Processing headers in parallel...")
    
    # Concurrent header processing using ThreadPoolExecutor for network-bound tasks.
    max_workers = 16
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_star, prefix, files, base_url): prefix
            for prefix, files in star_files.items()
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                star_data.append(result)

    if star_data:
        df = pd.DataFrame(star_data).dropna(subset=["RA", "DEC"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 40)
        print("       CATALOG INITIALIZATION SUMMARY")
        print("=" * 40)
        print(f"Successfully cataloged targets : {len(df)}")
        print(f"Output saved at location       : {output_path}")
        print("=" * 60 + "\n")
    else:
        print("[!] Critical Failure: Could not extract target metadata records from source.")

if __name__ == "__main__":
    raw_catalog_path = "data/raw/calspec_catalog.csv"
    create_catalog(raw_catalog_path)