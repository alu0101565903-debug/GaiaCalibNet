"""
Module: crossmatch.py
Report Section: 4.2 Gaia Data Retrieval & 4.3 Catalog Cross-matching

This script automates querying the Gaia DR3 database using coordinates from the
CALSPEC reference catalog. It uses a relaxed ADQL query to capture bright calibration
standards and white dwarfs, resolving epoch propagation (J2016.0 to J2000.0) and angular 
separation locally via Astropy.

Features auto-fallback to the Heidelberg GAVO TAP mirror if the ESA main archive is down.
"""

import os
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import TapPlus

# Silence astronomical warnings to keep the terminal output clean
warnings.filterwarnings("ignore")

def get_tap_client():
    """
    Checks the primary ESA Gaia archive and official European mirrors.
    Iterates through fallback servers until an active connection is established.
    """
    test_query = "SELECT TOP 1 source_id FROM gaiadr3.gaia_source"
    
    # List of official Gaia TAP mirrors
    tap_servers = [
        ("ESA Main", "https://gea.esac.esa.int/tap-server/tap"),
        ("ARI Heidelberg", "https://gaia.ari.uni-heidelberg.de/tap"),
        ("AIP Potsdam", "https://gaia.aip.de/tap"),
        ("INAF Italy", "https://gaia.inaf.it/tap")
    ]

    for name, url in tap_servers:
        print(f"-> Testing connection to {name}...")
        try:
            if name == "ESA Main":
                # Astroquery default Gaia object routes to ESA
                job = Gaia.launch_job(test_query, verbose=False)
                _ = job.get_results()
                print(f"[✓] {name} is ONLINE. Proceeding with the main server.")
                return Gaia
            else:
                # Use TapPlus for alternative mirrors
                tap_node = TapPlus(url=url)
                job = tap_node.launch_job(test_query, verbose=False)
                _ = job.get_results()
                print(f"[✓] Connected successfully to {name} mirror!")
                return tap_node
        except Exception as e:
            # Catch HTML maintenance responses and generic connection errors
            print(f"  [!] {name} is unreachable or in maintenance.")
            continue

    raise RuntimeError("Critical Error: ESA main server and all European backup mirrors are unreachable.")

def run_crossmatch(input_csv: str, output_csv: str, max_separation_arcsec: float = 5.0):
    """
    Performs a stable ADQL crossmatch with Gaia DR3, handling epoch propagation 
    and candidate filtering locally to avoid Gaia server timeout errors and to 
    preserve high proper-motion stars.
    
    Parameters:
        input_csv (str): Path to the input CALSPEC catalog CSV.
        output_csv (str): Path where the matched catalog will be saved.
        max_separation_arcsec (float): Maximum allowed angular separation in arcseconds 
                                       at epoch J2000.0 (default is 5.0").
    """
    print("=" * 60)
    print("RUNNING STABLE CROSS-MATCH: CALSPEC <-> GAIA DR3 (LOCAL EPOCH PROP)")
    print("=" * 60)
    
    if not os.path.exists(input_csv):
        print(f"[!] Critical Error: File '{input_csv}' does not exist.")
        return

    df_calspec = pd.read_csv(input_csv, encoding="utf-8-sig")
    # Strip whitespace from column names to avoid lookup errors
    df_calspec.columns = df_calspec.columns.str.strip()
    print(f"-> Loading {len(df_calspec)} targets from the CALSPEC catalog...")
    
    # Resolve and select the operational server (ESA or GAVO Heidelberg)
    try:
        tap_service = get_tap_client()
    except Exception as e:
        print(f"\n[!] {e}")
        return

    crossmatch_data = []

    for idx, row in df_calspec.iterrows():
        target_name = row.get("calspec_name", row.get("Name"))
        ra_calspec = row.get("ra_calspec", row.get("RA"))
        dec_calspec = row.get("dec_calspec", row.get("DEC"))
        fits_file = row.get("fits_file", row.get("File_Source"))
        
        if pd.isna(ra_calspec) or pd.isna(dec_calspec):
            continue

        # CALSPEC base coordinates at epoch J2000.0
        coord_calspec = SkyCoord(ra=ra_calspec, dec=dec_calspec, unit=(u.deg, u.deg), frame="icrs")

        # Simplified ADQL query:
        # 1. 30" search radius to account for proper motion over 16 years (J2000 to J2016).
        # 2. has_xp_continuous = true (strict boolean expression for PostgreSQL).
        # 3. Removed RUWE and G-mag restrictions to prevent dropping bright calibration standards.
        query = f"""
        SELECT 
            source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp, ruwe
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_calspec}, {dec_calspec}, {30.0 / 3600.0})
        )
        AND has_xp_continuous = 'true'
        """
        
        try:
            # Execute query on the active server instance
            job = tap_service.launch_job(query, verbose=False)
            results = job.get_results()
            
            if len(results) > 0:
                candidates = []
                
                for match in results:
                    # Clean proper motion and parallax fields (handle missing/masked values)
                    pmra_val = 0.0 if np.ma.is_masked(match["pmra"]) or np.isnan(match["pmra"]) else match["pmra"]
                    pmdec_val = 0.0 if np.ma.is_masked(match["pmdec"]) or np.isnan(match["pmdec"]) else match["pmdec"]
                    
                    plx_raw = match["parallax"]
                    plx_val = 0.0 if np.ma.is_masked(plx_raw) or np.isnan(plx_raw) or plx_raw <= 0 else plx_raw

                    # Build the Gaia source coordinate at the Gaia reference epoch J2016.0.
                    # If the parallax is invalid, omit the distance and only use proper motions.
                    coord_gaia_j2016 = SkyCoord(
                        ra = match["ra"] * u.deg, 
                        dec = match["dec"] * u.deg, 
                        distance=None if plx_val == 0 else (1000.0 / plx_val) * u.pc,
                        pm_ra_cosdec = pmra_val * u.mas / u.yr,
                        pm_dec = pmdec_val * u.mas / u.yr,
                        obstime = Time("J2016.0"),
                        frame = "icrs"
                    )
                    
                    # Propagate Gaia coordinates back to J2000.0 using proper motion.
                    # This enables a direct and accurate comparison with the CALSPEC catalog.
                    coord_gaia_j2000 = coord_gaia_j2016.apply_space_motion(new_obstime=Time("J2000.0"))

                    # Compute the physical angular separation at epoch J2000.0
                    angular_sep = coord_calspec.separation(coord_gaia_j2000).arcsec

                    # Validate candidate against the maximum separation tolerance
                    if angular_sep <= max_separation_arcsec:
                        candidates.append((angular_sep, match))
                
                if candidates:
                    # Sort candidates by minimum angular distance
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
                    print(f"  [✓] {target_name:<15} -> MATCHED at {best_sep:.3f}\" (G-mag: {best_match['phot_g_mean_mag']:.2f})")
                else:
                    print(f"  [X] {target_name:<15} -> Filtered out (No candidates within {max_separation_arcsec}\" at J2000.0)")
            else:
                print(f"  [X] {target_name:<15} -> No Gaia DR3 continuous XP counterpart found.")
                
        except Exception as e:
            print(f"  [!] Query failed for {target_name}: {e}")
            continue

    # Clean duplicates and finalize output table
    if crossmatch_data:
        df_final = pd.DataFrame(crossmatch_data)
        initial_len = len(df_final)
        
        # Keep only the closest Gaia match per source_id to avoid duplicate entries
        df_final = df_final.sort_values("angular_distance_arcsec").drop_duplicates(subset=["gaia_source_id"], keep="first")
        duplicate_count = initial_len - len(df_final)
        
        if duplicate_count > 0:
            print(f"\n[i] Resolved {duplicate_count} crossmatch conflicts via minimal angular distance routing.")

        # Ensure output directory exists before saving
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
    input_file = "data/raw/calspec_catalog.csv"
    output_file = "data/processed/crossmatch_catalog.csv"
    
    # Executing the pipeline with a 5.0 arcsec tolerance at J2000
    run_crossmatch(input_file, output_file, max_separation_arcsec=5.0)