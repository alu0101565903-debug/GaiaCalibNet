"""
Module: crossmatch.py
Report Section: 4.2 Gaia Data Retrieval & 4.3 Catalog Cross-matching

This script automates querying the Gaia DR3 database using coordinates from the
CALSPEC reference catalog. It queries Gaia DR3 using a standard, stable ADQL query,
and resolves epoch propagation (J2016.0 to J2000) and angular separation locally 
via Astropy to guarantee server stability.

Features auto-fallback to Heidelberg GAVO TAP mirror if ESA main archive is down.
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

# Silence astronomical warnings to keep the terminal clean
warnings.filterwarnings("ignore")

def get_tap_client():
    """
    Checks if ESA Gaia is responsive. If not, returns a connection to the
    Heidelberg GAVO TAP Mirror so the pipeline never halts.
    """
    print("-> Checking ESA Gaia main archive status...")
    test_query = "SELECT TOP 1 source_id FROM gaiadr3.gaia_source"
    try:
        # Test the primary ESA Gaia archive with a lightweight query.
        # This avoids running expensive crossmatch jobs before verifying server health.
        job = Gaia.launch_job(test_query, verbose=False)
        _ = job.get_results()
        print("[✓] ESA Gaia server is ONLINE. Proceeding with main server.")
        return Gaia
    except Exception:
        print("[!] ESA Gaia is down or under maintenance. Redirecting to Heidelberg GAVO Mirror...")
        try:
            # Connect to the GAVO Heidelberg TAP mirror as a fallback.
            # The mirror exposes the same Gaia ADQL schema and can handle the same queries.
            gavo_tap = TapPlus(url="http://gavo.ari.uni-heidelberg.de/tap")
            # Verify connectivity before attempting the full crossmatch.
            job = gavo_tap.launch_job(test_query, verbose=False)
            _ = job.get_results()
            print("[✓] Connected successfully to GAVO Heidelberg mirror!")
            return gavo_tap
        except Exception as e:
            raise RuntimeError(f"Critical: Both ESA and GAVO mirror are unreachable. Details: {e}")


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
    # Strip whitespace from column names to avoid lookup errors when the CSV has extra spaces.
    df_calspec.columns = df_calspec.columns.str.strip()
    print(f"-> Loading {len(df_calspec)} targets from CALSPEC catalog...")
    
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

        # CALSPEC base coordinates in J2000
        coord_calspec = SkyCoord(ra=ra_calspec, dec=dec_calspec, unit=(u.deg, u.deg), frame="icrs")

        # Build a stable ADQL query around the CALSPEC coordinates.
        # The search radius is 3 arcseconds in ICRS and the query filters for
        # Gaia sources with continuous XP spectra and a reliable RUWE value.
        query = f"""
        SELECT 
            source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp, ruwe
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_calspec}, {dec_calspec}, {3.0 / 3600.0})
        )
        AND has_xp_continuous = 'True'
        AND ruwe < 1.4
        AND phot_g_mean_mag > 3.0
        """
        
        try:
            # Querying the active server instance
            job = tap_service.launch_job(query, verbose=False)
            results = job.get_results()
            
            if len(results) > 0:
                candidates = []
                
                for match in results:
                    # Clean proper motions and parallax fields
                    pmra_val = 0.0 if np.ma.is_masked(match["pmra"]) or np.isnan(match["pmra"]) else match["pmra"]
                    pmdec_val = 0.0 if np.ma.is_masked(match["pmdec"]) or np.isnan(match["pmdec"]) else match["pmdec"]
                    
                    plx_raw = match["parallax"]
                    plx_val = 0.0 if np.ma.is_masked(plx_raw) or np.isnan(plx_raw) or plx_raw <= 0 else plx_raw

                    # Build the Gaia source coordinate at the Gaia reference epoch J2016.0.
                    # If parallax is invalid, omit the distance and only use proper motions.
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
                    # This enables a direct comparison with the CALSPEC catalog coordinates.
                    coord_gaia_j2000 = coord_gaia_j2016.apply_space_motion(new_obstime=Time("J2000.0"))

                    # Compute the angular separation between the original CALSPEC position
                    # and the propagated Gaia position at epoch J2000.
                    angular_sep = coord_calspec.separation(coord_gaia_j2000).arcsec

                    # Physical check: must be closer than tolerance limit
                    if angular_sep <= max_separation_arcsec:
                        candidates.append((angular_sep, match))
                
                if candidates:
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
                    print(f"  [X] {target_name:<15} -> Filtered out (No candidates within {max_separation_arcsec}\" at J2000)")
            else:
                print(f"  [X] {target_name:<15} -> No Gaia DR3 counterparts found.")
                
        except Exception as e:
            print(f"  [!] Query failed for {target_name}: {e}")
            continue

    # Clean duplicates and finalize output table
    if crossmatch_data:
        df_final = pd.DataFrame(crossmatch_data)
        initial_len = len(df_final)
        # Keep only the closest Gaia match per source_id to avoid duplicate entries.
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
    input_file = "data/raw/calspec_catalog.csv"
    output_file = "data/processed/crossmatch_catalog.csv"
    run_crossmatch(input_file, output_file)
