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
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits

# Ignore Astropy and Pandas runtime warnings for cleaner terminal reporting
warnings.filterwarnings('ignore')

def clean_target_name(name: str, prefix: str) -> str:
    """
    Cleans and standardizes raw target strings to maximize coordinate resolution 
    success rates when querying the SIMBAD/Sesame name resolvers.

    Parameters:
    - name (str): The raw target name extracted directly from the FITS header.
    - prefix (str): The fallback prefix parsed directly from the asset filename string.

    Returns:
    - str: A normalized astronomical star identifier string.
    """
    base_name = name.upper() if name else prefix.upper()
    
    # Regular expressions to enforce standard cross-matching nomenclatures
    base_name = re.sub(r'^WD(\d{4})_(\d{3,4})$', r'WD \1-\2', base_name)
    base_name = re.sub(r'^WDFS(\d{4})_(\d{2})$', r'WDFS \1-\2', base_name)
    base_name = re.sub(r'^NGC(\d+)_(\d+)$', r'NGC \1 \2', base_name)
    base_name = re.sub(r'^BD_?(\d+)D(\d+)$', r'BD+\1 \2', base_name)
    base_name = re.sub(r'^2M(\d{6,})$', r'2MASS J\1', base_name)
    
    # Standardize string dividers
    base_name = base_name.replace('_', ' ')
    base_name = re.sub(r'^(\d+)([A-Z]+)$', r'\1 \2', base_name)
    
    return base_name.strip()

def get_coords_sesame(star_name: str):
    """
    Queries the Sesame/SIMBAD gateway using Astropy to resolve SkyCoord coordinates
    by object identifier when local FITS headers lack spatial reference tags.

    Parameters:
    - star_name (str): The standardized name of the target stellar system.

    Returns: tuple (Right Ascension, Declination) in decimal degrees rounded to 5 decimals, 
    or (None, None) if lookup fails.
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

def create_catalog(output_path: str):
    """
    Scrapes the remote index, structures targets, handles observational file prioritization,
    resolves coordinates, and dumps the raw consolidated metadata table.

    Parameters:
    - output_path (str): Target CSV file path where the generated raw index will be saved.
    """
    print("=" * 60)
    print("[4.1] SCRA-PARSING AND GENERATING BASE CALSPEC CATALOG")
    print("=" * 60)
    
    # Official archive directory route for remote file streaming
    base_url = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
    
    try:
        print("-> Querying remote STScI directory index page...")
        req = requests.get(base_url, timeout=15)
        fits_files = re.findall(r'href="([^"]+\.fits)"', req.text)
    except Exception as e:
        print(f"[!] Critical Connection Error: Unable to access STScI server: {e}")
        return
    
    # Group file variants matching identical target star roots
    star_files = {}
    for f in fits_files:
        match = re.search(r'^(.+?)_(mod|stis|nic|stiswfcnic|stiswfc|faint)', f)
        if match:
            prefix = match.group(1)
            star_files.setdefault(prefix, []).append(f)
            
    star_data = []
    print(f"-> Found {len(star_files)} distinct target groups. Processing metadata headers...")
    
    for prefix, files in star_files.items():
        # Optimization heuristic: Prioritize observational datasets over theoretical model grids ('mod')
        best_file = sorted(files, key=lambda x: 0 if 'mod' not in x else 1)[0]
        
        try:
            # Stream and pull the remote FITS primary header records via URL connection
            header = fits.getheader(base_url + best_file)
            raw_name = header.get('TARGNAME', prefix)
            
            # Extract standard spatial coordinates fields
            ra = header.get('RA_TARG', header.get('RA'))
            dec = header.get('DEC_TARG', header.get('DEC'))
            
            # Trigger SIMBAD/Sesame resolution loop if header positions are blank or invalid
            if not is_valid_coord(ra) or not is_valid_coord(dec):
                cleaned_name = clean_target_name(raw_name, prefix)
                ra, dec = get_coords_sesame(cleaned_name)
                
            if ra is not None and dec is not None:
                star_data.append({
                    'calspec_name': raw_name,
                    'RA': float(ra),
                    'DEC': float(dec),
                    'fits_file': best_file
                })
        except Exception:
            # Silently skip unreadable headers or transient streaming connection timeouts
            continue

    if star_data:
        df = pd.DataFrame(star_data).dropna(subset=['RA', 'DEC'])
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
    # Standardized paths according to the project layout requirements (Section 3.2)
    RAW_CATALOG_PATH = "data/raw/calspec_catalog.csv"
    create_catalog(RAW_CATALOG_PATH)