"""Build a CALSPEC target catalog from the STScI archive.

The script parses remote FITS headers to extract target names and coordinates,
then resolves missing coordinates with SIMBAD/Sesame fallbacks.

Features
--------
- Extracts V magnitudes from the STScI index.
- Parses FITS headers concurrently.
- Handles sexagesimal and decimal coordinate representations.

Notes
-----
Report context: Section 4.1, CALSPEC Data Acquisition and Catalog Generation.
"""

import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import conf
from bs4 import BeautifulSoup

# Suppress Astropy and Pandas runtime warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# Enforce a global timeout for astropy remote FITS header requests
conf.remote_timeout = 10 

def parse_fits_coords(
    ra_val: Union[float, str, None], 
    dec_val: Union[float, str, None]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract and parse celestial coordinates into decimal degrees.

    Supports both decimal degrees (float/numeric string) and sexagesimal 
    strings (HH:MM:SS / DD:MM:SS) typical of legacy STScI FITS headers.

    Parameters
    ----------
    ra_val : float, str, or None
        Right Ascension value extracted from the FITS header.
    dec_val : float, str, or None
        Declination value extracted from the FITS header.

    Returns
    -------
    ra : float or None
        Right Ascension in decimal degrees rounded to 5 decimal places.
    dec : float or None
        Declination in decimal degrees rounded to 5 decimal places.

    Notes
    -----
    Primary fallback converts sexagesimal strings via `astropy.coordinates.SkyCoord` 
    assuming `hourangle` for RA and `degree` for DEC.
    """
    if ra_val is None or dec_val is None or pd.isna(ra_val) or pd.isna(dec_val) or ra_val == "" or dec_val == "":
        return None, None
        
    try:
        # First attempt: Direct decimal format (ideal)
        return float(ra_val), float(dec_val)
    except ValueError:
        # Second attempt: Sexagesimal string format (e.g., "15 14 21", "+00 12 33")
        try:
            coord = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg))
            return round(coord.ra.degree, 5), round(coord.dec.degree, 5)
        except Exception:
            return None, None

def clean_target_name(name: str, prefix: str) -> List[str]:
    """
    Clean raw target names and generate prioritized SIMBAD search candidates.

    Standardizes target names to avoid blind zero-padding mismatches during
    SIMBAD resolution while retaining the original CALSPEC identifier.

    Parameters
    ----------
    name : str
        The raw target name extracted from FITS header (`TARGNAME` or `OBJECT`).
    prefix : str
        Fallback identifier prefix derived from the filename.

    Returns
    -------
    candidates : list of str
        A prioritized list of string query candidates for SIMBAD resolution.

    Notes
    -----
    CALSPEC identifiers such as P330E, P177D, SNAP2, KF06T1, and C26202
    are resolved by their own names, never substituted with other standards.
    """
    base_name = (name or "").strip().upper() or prefix.strip().upper()
    candidates = [base_name] if base_name else []

    # CALSPEC standards must retain their own identities. A failed lookup is
    # preferable to assigning coordinates from an unrelated standard star.
    calspec_aliases = {
        "ETA1DOR": ["eta Dor"],
        "SUN": ["Sun"],
        "SUN_REFERENCE": ["Sun"]
    }
    
    if base_name in calspec_aliases:
        return calspec_aliases[base_name]

    # Resolve White Dwarf naming conventions (WD / WDFS)
    wd_match = re.match(r"^(?:WDFS|WD)_?(\d{4})[_\-](\d{2,4})$", base_name)
    if wd_match:
        candidates.append(f"WD {wd_match.group(1)}-{wd_match.group(2)}")
        candidates.append(f"WD {wd_match.group(1)}+{wd_match.group(2)}")
    
    cleaned = base_name.replace("_", " ").strip()
    if cleaned not in candidates:
        candidates.append(cleaned)

    # 2MASS string formatting
    tmass_match = re.match(r"^2M(\d{6,})$", base_name)
    if tmass_match:
        candidates.append(f"2MASS J{tmass_match.group(1)}")

    # Bonner Durchmusterung (BD) string formatting
    bd_match = re.match(r"^BD_?([PM\+\-])?(\d+)[D_]?(\d+)$", base_name)
    if bd_match:
        sign = bd_match.group(1)
        deg = bd_match.group(2)
        num = bd_match.group(3)
        
        if sign in ['P', '+']:
            candidates.append(f"BD+{deg} {num}")
        elif sign in ['M', '-']:
            candidates.append(f"BD-{deg} {num}")
        else:
            candidates.append(f"BD+{deg} {num}")
            candidates.append(f"BD-{deg} {num}")

    # New General Catalogue (NGC) formatting
    ngc_match = re.match(r"^NGC(\d+)[\s_]+(\d+)$", base_name)
    if ngc_match:
        candidates.append(f"NGC {ngc_match.group(1)} {ngc_match.group(2)}")

    return candidates

def get_coords_sesame(candidates: List[str])-> Tuple[Optional[float], Optional[float]]:
    """
    Query CDS/SIMBAD to resolve coordinates for a given list of name candidates.

    Parameters
    ----------
    candidates : list of str
        List of candidate target names to query iteratively via `SkyCoord.from_name`.

    Returns
    -------
    ra : float or None
        Rounded decimal degree Right Ascension on the first successful match.
    dec : float or None
        Rounded decimal degree Declination on the first successful match.
    """
    for cand in candidates:
        try:
            coord = SkyCoord.from_name(cand)
            return round(coord.ra.degree, 5), round(coord.dec.degree, 5)
        except Exception:
            continue
    return None, None

def get_fits_keyword(header: fits.Header, keys: List[str]) -> Optional[Union[str, float, int]]:
    """
    Extract the first valid non-empty value matching a list of potential FITS keys.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The FITS header container to query.
    keys : list of str
        Ordered list of candidate keyword strings to look up.

    Returns
    -------
    val : str, float, int, or None
        The value associated with the first matching key, or None if no valid key exists.
    """
    for k in keys:
        if k in header and header[k] is not None and str(header[k]).strip() != "":
            return header[k]
    return None

def process_single_star(prefix: str, files: list, base_url: str) -> Optional[Dict[str, Union[str, float]]]:
    """
    Process FITS headers for a single star target group to extract metadata and coordinates.

    Executes a multi-tier resolution strategy:
    1. Primary FITS extension (Extension 0) header parse.
    2. Secondary extension (Extension 1) header parse fallback.
    3. SIMBAD name resolution fallback.

    Parameters
    ----------
    prefix : str
        Unique target identifier prefix.
    files : list of str
        Associated FITS filenames for this target.
    base_url : str
        Base URL pointing to the remote STScI CALSPEC directory.

    Returns
    -------
    metadata : dict or None
        Metadata dictionary containing 'calspec_name', 'RA', 'DEC', 'fits_file', 
        and 'coord_source', or None if processing fails or target only has model fits.
    """
    valid_files = [f for f in files if "mod" not in f.lower()]
    
    if not valid_files:
        print(f"  [-]  {prefix}: Only theoretical models (mod).")
        return None
        
    best_file = sorted(valid_files)[0]
    coord_source = None

    try:
        url = base_url + best_file
        
        # Tier 1: Primary HDU (Extension 0)
        header = fits.getheader(url, ext=0)
        ra_raw = get_fits_keyword(header, ["RA_TARG", "RA", "TARG_RA", "RA_OBJ", "OBJ_RA"])
        dec_raw = get_fits_keyword(header, ["DEC_TARG", "DEC", "TARG_DEC", "DEC_OBJ", "OBJ_DEC"])
        raw_name = str(get_fits_keyword(header, ["TARGNAME", "OBJECT"]) or prefix).strip()
        
        ra, dec = parse_fits_coords(ra_raw, dec_raw)
        if ra is not None and dec is not None:
            coord_source = "FITS_EXT0"
        
        # Tier 2: Extension 1 Fallback
        if ra is None or dec is None:
            try:
                header_ext1 = fits.getheader(url, ext=1)
                ra_raw1 = get_fits_keyword(header_ext1, ["RA_TARG", "RA", "TARG_RA", "CRVAL1"])
                dec_raw1 = get_fits_keyword(header_ext1, ["DEC_TARG", "DEC", "TARG_DEC", "CRVAL2"])
                ra, dec = parse_fits_coords(ra_raw1, dec_raw1)
                
                if ra is not None and dec is not None:
                    coord_source = "FITS_EXT1"
            except Exception:
                pass
            
        # Tier 3: SIMBAD Sesame Fallback
        if ra is None or dec is None:
            candidates = clean_target_name(raw_name, prefix)
            ra, dec = get_coords_sesame(candidates)
            
            if ra is not None and dec is not None:
                coord_source = "SIMBAD_J2000"
            else:
                print(f"  [-] Rejected {prefix}: SIMBAD rejected alternatives: {candidates}")
                return None
                
        return {
            "calspec_name": raw_name,
            "RA": float(ra),
            "DEC": float(dec),
            "fits_file": best_file,
            "coord_source": coord_source
        }
    except Exception as e:
        print(f"  [!] Read failure for {prefix} ({best_file}): {str(e)}")
        
    return None

def fetch_stsci_v_magnitudes() -> Optional[pd.DataFrame]:
    """
    Scrape official STScI CALSPEC web tables to extract target optical 'V' magnitudes.

    Uses BeautifulSoup to decompose HTML footnote tags (sup), clean subheaders, 
    and reliably parse optical 'V' values across non-uniform table columns.

    Returns
    -------
    df_mags : pandas.DataFrame or None
        DataFrame containing target lookup keys and corresponding 'V' magnitudes, 
        or None if web scraping fails.
    """
    url = "https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec"
    print("\n-> Scraping official V magnitudes from STScI HTML tables...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove footnote superscripts (<sup>) so numbers like BD+28 4211⁴ become BD+28 4211
        for sup in soup.find_all('sup'):
            sup.decompose()

        records = []
        tables = soup.find_all('table')

        for tbl in tables:
            rows = tbl.find_all('tr')
            if not rows:
                continue

            header_idx = -1
            star_col, v_col, name_col = -1, -1, -1

            # Search top rows for header positions
            for idx, row in enumerate(rows[:3]):
                cols = [td.get_text(strip=True).lower() for td in row.find_all(['th', 'td'])]
                for c_idx, c_text in enumerate(cols):
                    if 'star' in c_text and 'name' in c_text:
                        star_col = c_idx
                    elif c_text == 'v' or c_text == 'v mag':
                        v_col = c_idx
                    elif c_text == 'name':
                        name_col = c_idx

                if star_col != -1 and v_col != -1:
                    header_idx = idx
                    break

            if header_idx == -1:
                continue

            # Parse table data rows
            for row in rows[header_idx + 1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                if len(cells) <= max(star_col, v_col):
                    continue

                raw_star = cells[star_col] if star_col < len(cells) else ""
                raw_v = cells[v_col] if v_col < len(cells) else ""
                raw_name = cells[name_col] if (name_col != -1 and name_col < len(cells)) else ""

                # Ignore subheaders like [1], [2], [3]
                if raw_v.startswith('[') or raw_star.startswith('['):
                    continue

                # Regex extract floats (handles "-26.75", "12.27", "G=18.66")
                v_match = re.search(r'[-+]?\d*\.?\d+', raw_v)
                if not v_match:
                    continue
                
                try:
                    v_val = float(v_match.group(0))
                except ValueError:
                    continue

                # Add mapping keys (removing all spaces and special chars for maximum resilience)
                if raw_star:
                    key_star = re.sub(r'[^A-Z0-9]', '', raw_star.upper())
                    if key_star:
                        records.append({'key': key_star, 'V': v_val})

                if raw_name:
                    key_name = re.sub(r'[^A-Z0-9]', '', raw_name.upper())
                    if key_name:
                        records.append({'key': key_name, 'V': v_val})

        if records:
            df_mags = pd.DataFrame(records).drop_duplicates(subset=['key'])
            print(f"-> Successfully scraped {len(df_mags)} magnitude entries from STScI tables.")
            return df_mags

    except Exception as e:
        print(f"  [!] Web scrape error: {e}")

    return None

def create_catalog(output_path: str) -> None:
    """
    Execute catalog generation pipeline: scrape indices, extract headers, and export CSV.

    Parameters
    ----------
    output_path : str
        Path where the final consolidated CALSPEC metadata CSV should be written.
    """
    print("=" * 60)
    print("[4.1] SCRAPING AND GENERATING THE BASE CALSPEC CATALOG")
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
    
    star_files = {}
    for f in fits_files:
        match = re.search(r"^(.+?)_(mod|stis|nic|stiswfcnic|stiswfc|faint|spec|fos|acs|wfc3|wfpc2|iue)(?:_\w+)?\.fits$", f, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            star_files.setdefault(prefix, []).append(f)

    star_data = []
    total_groups = len(star_files)
    
    if total_groups == 0:
        print("[!] Warning: Found 0 target groups. Check STScI HTML format.")
        return

    print(f"-> Found {total_groups} distinct target groups. Processing headers in parallel...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
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

        df_mags = fetch_stsci_v_magnitudes()
        if df_mags is not None:
            mag_dict = dict(zip(df_mags['key'], df_mags['V']))

            def match_v(row):
                # 1. Match by CALSPEC name.
                k1 = re.sub(r'[^A-Z0-9]', '', str(row['calspec_name']).upper())
                if k1 in mag_dict:
                    return mag_dict[k1]

                # 2. Match by filename prefix, such as 'bd_28d4211' in 'bd_28d4211_stis_001.fits'.
                fits_base = os.path.basename(str(row['fits_file']))
                prefix_match = re.search(r"^(.+?)_(mod|stis|nic|stiswfcnic|stiswfc|faint|spec|fos|acs|wfc3|wfpc2|iue)", fits_base, re.IGNORECASE)
                if prefix_match:
                    k2 = re.sub(r'[^A-Z0-9]', '', prefix_match.group(1).upper())
                    if k2 in mag_dict:
                        return mag_dict[k2]

                return np.nan

            df['V'] = df.apply(match_v, axis=1)
            assigned_mags = df['V'].notna().sum()
            print(f"-> Successfully assigned official V magnitudes to {assigned_mags}/{len(df)} targets.")
        else:
            df['V'] = np.nan

        output_dir = os.path.dirname(output_path)
        if output_dir:  
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 40)
        print("        CATALOG INITIALIZATION SUMMARY")
        print("=" * 40)
        print(f"Successfully cataloged targets : {len(df)}")
        print(f"Output saved at location       : {output_path}")
        print("=" * 60 + "\n")
    else:
        print("[!] Critical Failure: Could not extract target metadata records.")

if __name__ == "__main__":
    raw_catalog_path = "data/raw/calspec_catalog.csv"
    create_catalog(raw_catalog_path)