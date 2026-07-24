"""
Module: create_catalog.py
Report Section: 4.1 CALSPEC Data Acquisition & Catalog Generation

This script scrapes the remote STScI CALSPEC server index, organizes files by target,
parses remote FITS headers stream-on-the-fly to extract target names and coordinates,
and leverages a multi-candidate SIMBAD/Sesame fallback parser to resolve missing 
coordinates, especially optimized for rescuing White Dwarfs (WD / WDFS).
"""

import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

# Ignore Astropy and Pandas runtime warnings for cleaner terminal reporting
warnings.filterwarnings("ignore")

def parse_fits_coords(ra_val, dec_val):
    """
    Try to extract coordinates, supporting both decimal degrees (float)
    and sexagesimal strings (HH:MM:SS / DD:MM:SS) typical of STScI FITS files.
    """
    if ra_val is None or dec_val is None or pd.isna(ra_val) or pd.isna(dec_val) or ra_val == "" or dec_val == "":
        return None, None
        
    try:
        # First attempt: Direct decimal format (ideal)
        return float(ra_val), float(dec_val)
    except ValueError:
        # Second attempt: Sexagesimal string format (e.g., "15 14 21", "+00 12 33")
        try:
            # Astropy assumes RA in hours and DEC in degrees by default for sexagesimal
            coord = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg))
            return round(coord.ra.degree, 5), round(coord.dec.degree, 5)
        except Exception:
            return None, None

def clean_target_name(name: str, prefix: str) -> list:
    """
    Clean and standardize target names, generating a LIST of candidate names for SIMBAD queries,
    optimized to rescue White Dwarfs (WD / WDFS) and other problematic objects.
    """
    base_name = name.upper() if name else prefix.upper()
    candidates = []

    # Alias dictionary for famous CALSPEC stars that SIMBAD may not recognize
    calspec_aliases = {
        "P330E": ["BD+28 4211"],
        "P177D": ["BD+25 4655"],
        "P041C": ["BD+20 5000"],
        "SNAP2": ["HD 107513"],
        "KF06T1": ["BD+28 4211"], # Muchas KF son calibradores cruzados
        "C26202": ["HD 188350"],
        "ETA1DOR": ["eta Dor"]
    }
    
    if base_name in calspec_aliases:
        return calspec_aliases[base_name]

    # Standardize WDFS and WD to the same base format "WD XXXX-YYY"
    base_name = re.sub(r"^WDFS_?(\d{4})_?(\d{2})$", r"WD \1-\2", base_name)
    base_name = re.sub(r"^WD_?(\d{4})_?(\d{2,4})$", r"WD \1-\2", base_name)
    
    cleaned = base_name.replace("_", " ").strip()
    candidates.append(cleaned)

    # If it's a White Dwarf, generate the battery of syntactic variations
    wd_match = re.match(r"^WD (\d{4})-(\d+)$", cleaned)
    if wd_match:
        ra_p, dec_p = wd_match.groups()
        candidates.append(f"WD {ra_p}-{dec_p}0")   # Fill with a 0 (e.g., WD 0148-4670)
        candidates.append(f"WD {ra_p}-{dec_p}00")  # Fill with two 0s
        candidates.append(f"WD{ra_p}-{dec_p}")     # Without space
        candidates.append(f"WD {ra_p}+{dec_p}")    # Positive sign just in case
        
    # Add variations for truncated 2MASS names
    tmass_match = re.match(r"^2M(\d{6,})$", base_name)
    if tmass_match:
        candidates.append(f"2MASS J{tmass_match.group(1)}")
        
    # Add variations for BD names
    bd_match = re.match(r"^BD_?(\d+)D(\d+)$", base_name)
    if bd_match:
        candidates.append(f"BD+{bd_match.group(1)} {bd_match.group(2)}")

    # Add variations for NGC names
    ngc_match = re.match(r"^NGC(\d+)_(\d+)$", base_name)
    if ngc_match:
        candidates.append(f"NGC {ngc_match.group(1)} {ngc_match.group(2)}")

    return candidates

def get_coords_sesame(candidates: list):
    """
    Iterate over the generated candidate list and query SIMBAD.
    Stops and returns coordinates on the first successful match.
    """
    for cand in candidates:
        try:
            coord = SkyCoord.from_name(cand)
            return round(coord.ra.degree, 5), round(coord.dec.degree, 5)
        except Exception:
            continue
    return None, None

def get_fits_keyword(header, keys: list):
    """Iterate over a list of possible FITS keys and return the first valid value."""
    for k in keys:
        if k in header and header[k] is not None and str(header[k]).strip() != "":
            return header[k]
    return None

def process_single_star(prefix: str, files: list, base_url: str) -> dict:
    """Worker function to process a single star group."""
    valid_files = [f for f in files if "mod" not in f.lower()]
    
    if not valid_files:
        print(f"  [-]  {prefix}: Solo modelos teóricos (mod).")
        return None
        
    best_file = sorted(valid_files)[0]

    try:
        url = base_url + best_file
        
        # 1. Intentar Extensión 0 (Primary HDU)
        header = fits.getheader(url, ext=0)
        ra_raw = get_fits_keyword(header, ["RA_TARG", "RA", "TARG_RA", "RA_OBJ", "OBJ_RA"])
        dec_raw = get_fits_keyword(header, ["DEC_TARG", "DEC", "TARG_DEC", "DEC_OBJ", "OBJ_DEC"])
        raw_name = str(get_fits_keyword(header, ["TARGNAME", "OBJECT"]) or prefix).strip()
        
        ra, dec = parse_fits_coords(ra_raw, dec_raw)
        
        # 2. Si falló la Extensión 0, intentar Extensión 1 de forma segura
        if ra is None or dec is None:
            try:
                header_ext1 = fits.getheader(url, ext=1)
                ra_raw1 = get_fits_keyword(header_ext1, ["RA_TARG", "RA", "TARG_RA", "CRVAL1"])
                dec_raw1 = get_fits_keyword(header_ext1, ["DEC_TARG", "DEC", "TARG_DEC", "CRVAL2"])
                # Importante: Volvemos a parsear las coordenadas leídas de ext=1
                ra, dec = parse_fits_coords(ra_raw1, dec_raw1)
            except Exception:
                # Si el FITS no tiene extensión 1, ignoramos el fallo y continuamos a SIMBAD
                pass
            
        # 3. Si las cabeceras FITS no tenían coordenadas válidas, usar SIMBAD multi-candidato
        if ra is None or dec is None:
            candidates = clean_target_name(raw_name, prefix)
            ra, dec = get_coords_sesame(candidates)
            
            if ra is None or dec is None:
                print(f"  [-] Rejected {prefix}: Invalid coordinates and SIMBAD rejected alternatives: {candidates}")
                return None
                
        return {
            "calspec_name": raw_name,
            "RA": float(ra),
            "DEC": float(dec),
            "fits_file": best_file
        }
    except Exception as e:
        print(f"  [!] Red or reading failure for {prefix} ({best_file}): {str(e)}")
        
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
        
        raw_hrefs = re.findall(r'href=["\']([^"\']+)["\']', req.text, re.IGNORECASE)
        fits_files = [os.path.basename(f) for f in raw_hrefs if f.lower().endswith(".fits")]
        
    except Exception as e:
        print(f"[!] Critical Connection Error: Unable to access STScI server: {e}")
        return
    
    # Group file variants matching identical target star roots.
    star_files = {}
    for f in fits_files:
        match = re.search(r"^(.+?)_(mod|stis|nic|stiswfcnic|stiswfc|faint|spec|fos|acs|wfc3|wfpc2|iue)(?:_\w+)?\.fits$", f, re.IGNORECASE)
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
    max_workers = 5
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