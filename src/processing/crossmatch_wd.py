"""
Module: crossmatch_wd.py
Report Section: 4.2 Gaia Data Retrieval & 4.3 Catalog Cross-matching

This script automates querying the Gaia DR3 database using coordinates from the
CALSPEC reference catalog via an asynchronous cone search. In cases of multiple 
sources within the search radius, it selects the brightest object in the G-band.
"""

import os
import warnings
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

# Silence astronomical warnings to keep the terminal clean
warnings.filterwarnings('ignore')

def run_crossmatch(input_csv: str, output_csv: str, radius_arcsec: float = 3.0):
    """
    Performs an asynchronous cone search crossmatch between the local CALSPEC 
    catalog and the Gaia DR3 archive.
    
    Parameters:
    - input_csv (str): Path to the raw CALSPEC catalog (must contain RA and DEC columns).
    - output_csv (str): Path where the consolidated crossmatched catalog will be saved.
    - radius_arcsec (float): Search radius in arcseconds for the cone search.
    """
    print("=" * 60)
    print("RUNNING CATALOG CROSS-MATCH: CALSPEC <-> GAIA DR3")
    print("=" * 60)
    
    if not os.path.exists(input_csv):
        print(f"[!] Critical Error: File {input_csv} does not exist.")
        print("Please ensure the base CALSPEC catalog has been downloaded or generated.")
        return

    # 1. Failsafe reading: remove BOM (invisible characters) and strip column whitespaces
    df_calspec = pd.read_csv(input_csv, encoding='utf-8-sig')
    df_calspec.columns = df_calspec.columns.str.strip()
    
    print(f"-> Loading {len(df_calspec)} targets from CALSPEC catalog...")
    
    crossmatch_data = []
    
    # 2. Ensure Gaia DR3 is the default query table
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" 
    
    for idx, row in df_calspec.iterrows():
        # 3. Dynamic naming: look for the new columns, fallback to old ones if necessary
        target_name = row.get('calspec_name', row.get('Name'))
        ra = row.get('ra_calspec', row.get('RA'))
        dec = row.get('dec_calspec', row.get('DEC'))
        fits_file = row.get('fits_file', row.get('File_Source'))
        
        # Safety filter in case any row slipped through without coordinates
        if pd.isna(ra) or pd.isna(dec):
            print(f"  [!] Skipping {target_name}: Missing coordinates (NaN).")
            continue

        try:
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
            
            # Asynchronous TAP query to the Gaia archive
            job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec, verbose=False)
            results = job.get_results()
            
            if len(results) > 0:
                # If there's more than one source within the radius, pick the brightest in G-band
                best_match = results[results['phot_g_mean_mag'].argmin()]
                
                crossmatch_data.append({
                    'calspec_name': target_name,
                    'ra_calspec': ra,
                    'dec_calspec': dec,
                    'fits_file': fits_file,
                    'gaia_source_id': str(best_match['source_id']),
                    'ra_gaia': best_match['ra'],
                    'dec_gaia': best_match['dec'],
                    'parallax': best_match['parallax'],
                    'pmra': best_match['pmra'],
                    'pmdec': best_match['pmdec'],
                    'phot_g_mean_mag': best_match['phot_g_mean_mag'],
                    'bp_rp': best_match['bp_rp']
                })
                print(f"  [✓] {target_name:<15} -> MATCHED (G-mag: {best_match['phot_g_mean_mag']:.2f})")
            else:
                print(f"  [X] {target_name:<15} -> No Gaia DR3 counterpart found within {radius_arcsec}\"")
                
        except Exception as e:
            print(f"  [!] Query failed for {target_name}: {e}")
            continue

    # 4. Save the results to the final CSV
    if crossmatch_data:
        df_final = pd.DataFrame(crossmatch_data)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_final.to_csv(output_csv, index=False)
        
        print("\n" + "=" * 60)
        print(f"-> Crossmatch Complete!")
        print(f"-> Total targets matched with Gaia DR3: {len(df_final)}")
        print(f"-> Output saved at: {output_csv}")
        print("=" * 60)
    else:
        print("\n[!] Critical Error: No stars were found in Gaia DR3.")

if __name__ == "__main__":
    # Standardized paths according to the project layout (Report Section 3.2)
    input_file = "data/raw/calspec_catalog.csv"
    output_file = "data/processed/crossmatch_catalog.csv"

    run_crossmatch(input_file, output_file)