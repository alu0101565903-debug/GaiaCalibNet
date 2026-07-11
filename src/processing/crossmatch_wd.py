"""
Module: crossmatch_wd.py
Report Section: 4.2 Gaia Data Retrieval & 4.3 Catalog Cross-matching

This script automates querying the Gaia DR3 database using coordinates from the
CALSPEC reference catalog. It queries Gaia DR3 using a standard, stable ADQL query,
and resolves epoch propagation (J2016.0 to J2000) and angular separation locally 
via Astropy to guarantee server stability.
"""

import os
import warnings
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia

# Silence astronomical warnings to keep the terminal clean
warnings.filterwarnings("ignore")

def run_crossmatch(input_csv: str, output_csv: str, max_separation_arcsec: float = 1.0):
    """
    Performs a stable ADQL crossmatch with Gaia DR3, handling epoch propagation 
    and candidate filtering locally to avoid Gaia server 500 errors.
    """
    print("=" * 60)
    print("RUNNING STABLE CROSS-MATCH: CALSPEC <-> GAIA DR3 (LOCAL EPOCH PROP)")
    print("=" * 60)
    
    if not os.path.exists(input_csv):
        print(f"[!] Critical Error: File {input_csv} does not exist.")
        return

    df_calspec = pd.read_csv(input_csv, encoding="utf-8-sig")
    df_calspec.columns = df_calspec.columns.str.strip()
    print(f"-> Loading {len(df_calspec)} targets from CALSPEC catalog...")
    
    crossmatch_data = []

    for idx, row in df_calspec.iterrows():
        target_name = row.get("calspec_name", row.get("Name"))
        ra_calspec = row.get("ra_calspec", row.get("RA"))
        dec_calspec = row.get("dec_calspec", row.get("DEC"))
        fits_file = row.get("fits_file", row.get("File_Source"))
        
        if pd.isna(ra_calspec) or pd.isna(dec_calspec):
            continue

        # CALSPEC base coordinates in J2000
        coord_calspec = SkyCoord(ra=ra_calspec, dec=dec_calspec, unit=(u.deg, u.deg), frame='icrs')

        # ULTRA-STABLE ADQL QUERY:
        # Search within a 3" radius in J2016.0 and filter for the presence of continuous XP
        query = f"""
        SELECT 
            source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp, ruwe
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_calspec}, {dec_calspec}, {3.0 / 3600.0})
        )
        AND has_xp_continuous = "True"
        AND ruwe < 1.4
        """
        
        try:
            # Launch the asynchronous job
            job = Gaia.launch_job_async(query, verbose=False)
            results = job.get_results()
            
            if len(results) > 0:
                candidates = []
                
                for match in results:
                    # Fill missing proper motions with 0 for stability
                    pmra_val = 0.0 if np.isnan(match["pmra"]) else match["pmra"]
                    pmdec_val = 0.0 if np.isnan(match["pmdec"]) else match["pmdec"]
                    plx_val = 0.0 if np.isnan(match["parallax"]) or match["parallax"] <= 0 else match["parallax"]

                    # Create the Gaia coordinate in its native epoch (DR3 = J2016.0), including proper motions
                    coord_gaia_j2016 = SkyCoord(
                        ra = match["ra"] * u.deg, 
                        dec = match["dec"] * u.deg, 
                        distance=None if plx_val == 0 else (1000.0 / plx_val) * u.pc,
                        pm_ra_cosdec = pmra_val * u.mas / u.yr,
                        pm_dec = pmdec_val * u.mas / u.yr,
                        obstime = Time("J2016.0"),
                        frame = "icrs"
                    )
                    
                    # Propagate the Gaia star backward in time (from J2016.0 to J2000.0)
                    coord_gaia_j2000 = coord_gaia_j2016.apply_space_motion(new_obstime=Time("J2000.0"))

                    # Measure the true angular separation in the common epoch J2000
                    angular_sep = coord_calspec.separation(coord_gaia_j2000).arcsec

                    # Strict threshold filter (<= max_separation_arcsec)
                    if angular_sep <= max_separation_arcsec:
                        candidates.append((angular_sep, match))
                
                if candidates:
                    # Choose the physically closest candidate as the match
                    candidates.sort(key=lambda x: x[0])
                    best_sep, best_match = candidates[0]
                    
                    crossmatch_data.append({
                        "calspec_name": target_name,
                        "ra_calspec": ra_calspec,
                        "dec_calspec": dec_calspec,
                        "fits_file": fits_file,
                        "gaia_source_id": str(best_match["source_id"]),
                        "ra_gaia": best_match["ra"],
                        "dec_gaia": best_match["dec"],
                        "angular_distance_arcsec": best_sep,
                        "parallax": best_match["parallax"],
                        "pmra": best_match["pmra"],
                        "pmdec": best_match["pmdec"],
                        "phot_g_mean_mag": best_match["phot_g_mean_mag"],
                        "bp_rp": best_match["bp_rp"],
                        "ruwe": best_match["ruwe"]
                    })
                    print(f"  [✓] {target_name:<15} -> MATCHED at {best_sep:.3f}\" (G-mag: {best_match["phot_g_mean_mag"]:.2f})")
                else:
                    print(f"  [X] {target_name:<15} -> Filtered out (No candidates within {max_separation_arcsec}\" at J2000)")
            else:
                print(f"  [X] {target_name:<15} -> No Gaia DR3 counterparts found.")
                
        except Exception as e:
            print(f"  [!] Query failed for {target_name}: {e}")
            continue

    # Remove duplicate Gaia source IDs
    if crossmatch_data:
        df_final = pd.DataFrame(crossmatch_data)
        initial_len = len(df_final)
        df_final = df_final.sort_values("angular_distance_arcsec").drop_duplicates(subset=["gaia_source_id"], keep="first")
        duplicate_count = initial_len - len(df_final)
        
        if duplicate_count > 0:
            print(f"\n[i] Resolved {duplicate_count} crossmatch conflicts via minimal angular distance routing.")

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_final.to_csv(output_csv, index=False)
        
        print("\n" + "=" * 60)
        print(f"-> Crossmatch Pipeline Success!")
        print(f"-> Total clean catalog size: {len(df_final)} samples (All with XP Spectra)")
        print(f"-> Output saved at: {output_csv}")
        print("=" * 60)
    else:
        print("\n[!] Critical Error: Zero stars survived the pipeline filters.")

if __name__ == "__main__":
    import numpy as np
    input_file = "data/raw/calspec_catalog.csv"
    output_file = "data/processed/crossmatch_catalog.csv"
    run_crossmatch(input_file, output_file)