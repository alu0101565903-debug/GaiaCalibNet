"""
Module: crossmatch.py
Report Section: 4.2 Gaia Data Retrieval & 4.3 Catalog Cross-matching

This module provides routines to cross-match a local CALSPEC stellar catalog with 
Gaia DR3. It performs cone searches via TAP queries, filters candidate matches using 
differential magnitudes (|G - V|), and accounts for proper motion by propagating 
astrometric coordinates from Gaia epoch J2016.0 to CALSPEC epoch J2000.0.

Features:
    - Multi-server TAP fallback (ESA Main, ARI Heidelberg, AIP Potsdam, INAF).
    - Strict angular distance filtering (default 1.5 - 5.0 arcsec).
    - Differential magnitude validation between Gaia G and CALSPEC V.
    - 3D space-motion propagation using Astropy SkyCoord and Time.
"""

import os
import warnings
from typing import Any, Dict, Union, List, Union

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.utils.tap.core import TapPlus

# Suppress Astropy and Pandas runtime warnings for cleaner terminal output
warnings.filterwarnings("ignore")


def get_tap_client() -> Union[Any, TapPlus]:
    """
    Test connectivity and retrieve an active TAP client from primary or mirror Gaia servers.

    Iterates through known public Gaia TAP service endpoints and verifies functionality 
    by executing a minimal query (`SELECT TOP 1`).

    Returns
    -------
    client : astroquery.gaia.Gaia or astroquery.utils.tap.core.TapPlus
        An active, responsive TAP connection instance.

    Raises
    ------
    RuntimeError
        If the primary ESA server and all backup mirror services fail to respond.

    Notes
    -----
    Tested mirrors include ESA Main, ARI Heidelberg, AIP Potsdam, and INAF Italy.
    """
    test_query = "SELECT TOP 1 source_id FROM gaiadr3.gaia_source"
    tap_servers = [
        ("ESA Main", "https://gea.esac.esa.int/tap-server/tap"),
        ("ARI Heidelberg", "https://gaia.ari.uni-heidelberg.de/tap"),
        ("AIP Potsdam", "https://gaia.aip.de/tap"),
        ("INAF Italy", "https://gaia.inaf.it/tap"),
    ]

    for name, url in tap_servers:
        print(f"-> Testing connection to {name}...")
        try:
            if name == "ESA Main":
                job = Gaia.launch_job(test_query, verbose=False)
                _ = job.get_results()
                print(f"[✓] {name} is ONLINE. Proceeding with main server.")
                return Gaia
            else:
                tap_node = TapPlus(url=url)
                job = tap_node.launch_job(test_query, verbose=False)
                _ = job.get_results()
                print(f"[✓] Connected successfully to {name} mirror!")
                return tap_node
        except Exception:
            print(f"  [!] {name} is unreachable or in maintenance.")
            continue

    raise RuntimeError("Critical Error: ESA main server and backup mirrors are unreachable.")


def run_crossmatch(
    input_csv: str,
    output_csv: str,
    max_separation_arcsec: float = 5.0,
    max_mag_diff: float = 0.3
) -> None:
    """
    Cross-match a CALSPEC catalog with Gaia DR3 using position and magnitude criteria.

    Parameters
    ----------
    input_csv : str
        File path to the input CALSPEC CSV catalog.
    output_csv : str
        File path where the output cross-matched CSV dataset will be saved.
    max_separation_arcsec : float, optional
        Maximum allowed angular separation in arcseconds between target and Gaia source.
        Default is 1.5.
    max_mag_diff : float, optional
        Maximum allowable absolute difference between Gaia phot_g_mean_mag and 
        CALSPEC optical V magnitude (|G - V|). Default is 1.5.

    Returns
    -------
    None

    Notes
    -----
    - Queries `gaiadr3.gaia_source` requiring `has_xp_continuous = 'True'`.
    - Converts Gaia source coordinates from reference epoch J2016.0 to epoch J2000.0 
      via 3D space-motion transformation (`apply_space_motion`) when proper motion 
      and parallax are available.
    """
    print("=" * 60)
    print("RUNNING STRICT MAGNITUDE CROSS-MATCH: CALSPEC <-> GAIA DR3")
    print("=" * 60)

    if not os.path.exists(input_csv):
        print(f"[!] Critical Error: File '{input_csv}' does not exist.")
        return

    df_calspec = pd.read_csv(input_csv, encoding="utf-8-sig", dtype=str)
    df_calspec.columns = df_calspec.columns.str.strip()
    print(f"-> Loading {len(df_calspec)} targets from CALSPEC catalog...")

    # Identify optical V magnitude column variations
    mag_cols = ["V", "Vmag", "v_mag", "m_V", "Magnitude", "mag"]
    v_col = next((col for col in mag_cols if col in df_calspec.columns), None)

    if not v_col:
        print("[!] Warning: No V magnitude column found in input CSV.")
        print("    Running cross-match without differential magnitude validation.")

    try:
        tap_service = get_tap_client()
    except Exception as e:
        print(f"\n[!] {e}")
        return

    crossmatch_data: List[Dict[str, Any]] = []

    for idx, row in df_calspec.iterrows():
        target_name = row.get("calspec_name", row.get("Name"))
        ra_calspec = row.get("ra_calspec", row.get("RA", row.get("ra")))
        dec_calspec = row.get("dec_calspec", row.get("DEC", row.get("dec")))
        fits_file = row.get("fits_file", row.get("File_Source"))

        ref_mag = float(row[v_col]) if v_col and pd.notna(row[v_col]) else None

        if pd.isna(ra_calspec) or pd.isna(dec_calspec):
            continue

        coord_calspec = SkyCoord(
            ra=ra_calspec, 
            dec=dec_calspec, 
            unit=(u.deg, u.deg), 
            frame="icrs"
        )

        # TAP Query: Retrieve Gaia DR3 sources within a 15 arcsecond cone radius
        query = f"""
        SELECT 
            source_id, ra, dec, parallax, pmra, pmdec, radial_velocity, phot_g_mean_mag, bp_rp, ruwe
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_calspec}, {dec_calspec}, {15.0 / 3600.0})
        ) = 1
        AND has_xp_continuous = 'True'
        """

        try:
            job = tap_service.launch_job(query, verbose=False)
            results = job.get_results()

            if len(results) > 0:
                candidates = []

                for match in results:
                    g_mag = match["phot_g_mean_mag"]

                    # Exclude invalid magnitude entries
                    if np.ma.is_masked(g_mag) or np.isnan(g_mag):
                        continue

                    # Differential Magnitude Filter
                    if ref_mag is not None:
                        mag_diff = abs(g_mag - ref_mag)
                        if mag_diff > max_mag_diff:
                            continue

                    # Astrometric parameter cleanup
                    pmra_val = 0.0 if np.ma.is_masked(match["pmra"]) or np.isnan(match["pmra"]) else float(match["pmra"])
                    pmdec_val = 0.0 if np.ma.is_masked(match["pmdec"]) or np.isnan(match["pmdec"]) else float(match["pmdec"])
                    rv_val = 0.0 if np.ma.is_masked(match["radial_velocity"]) or np.isnan(match["radial_velocity"]) else float(match["radial_velocity"])
                    
                    plx_raw = match["parallax"]
                    plx_val = 0.0 if np.ma.is_masked(plx_raw) or np.isnan(plx_raw) or plx_raw <= 0 else float(plx_raw)

                    # Epoch Propagation from J2016.0 to J2000.0
                    coord_gaia_j2016 = SkyCoord(
                        ra=match["ra"] * u.deg,
                        dec=match["dec"] * u.deg,
                        distance=None if plx_val == 0 else (1000.0 / plx_val) * u.pc,
                        pm_ra_cosdec=pmra_val * u.mas / u.yr,
                        pm_dec=pmdec_val * u.mas / u.yr,
                        radial_velocity=rv_val * u.km / u.s,
                        obstime=Time("J2016.0"),
                        frame="icrs",
                    )
                    coord_gaia_j2000 = coord_gaia_j2016.apply_space_motion(new_obstime=Time("J2000.0"))
                    angular_sep = coord_calspec.separation(coord_gaia_j2000).arcsec
                
                    if angular_sep <= max_separation_arcsec:
                        candidates.append((angular_sep, match))

                if candidates:
                    # Sort candidates by angular separation and select nearest source
                    candidates.sort(key=lambda x: x[0])
                    best_sep, best_match = candidates[0]
                    g_mag = best_match["phot_g_mean_mag"]

                    crossmatch_data.append({
                        "calspec_name": target_name,
                        "ra_calspec": ra_calspec,
                        "dec_calspec": dec_calspec,
                        "fits_file": fits_file,
                        "calspec_v_mag": ref_mag,
                        "gaia_source_id": str(best_match["source_id"]),
                        "ra_gaia": best_match["ra"],
                        "dec_gaia": best_match["dec"],
                        "angular_distance_arcsec": best_sep,
                        "parallax": best_match["parallax"],
                        "pmra": best_match["pmra"],
                        "pmdec": best_match["pmdec"],
                        "phot_g_mean_mag": g_mag,
                        "bp_rp": best_match["bp_rp"],
                        "ruwe": best_match["ruwe"],
                    })

                    print(
                        f"  [✓] {target_name:<15} -> MATCHED! "
                        f"Dist: {best_sep:.2f}\" | CALSPEC V: {ref_mag} | Gaia G: {g_mag:.2f}"
                    )
                else:
                    if ref_mag is not None:
                        print(
                            f"  [X] {target_name:<15} -> Rejected "
                            f"(No targets with |G - V| <= {max_mag_diff} & sep <= {max_separation_arcsec}\")"
                        )
                    else:
                        print(f"  [X] {target_name:<15} -> Rejected (No targets <= {max_separation_arcsec}\")")
            else:
                print(f"  [X] {target_name:<15} -> No Gaia DR3 continuous XP counterpart found.")

        except Exception as e:
            print(f"  [!] Query failed for {target_name}: {e}")
            continue

    # Clean duplicates and export to disk
    if crossmatch_data:
        df_final = pd.DataFrame(crossmatch_data)
        
        df_final = df_final.sort_values("angular_distance_arcsec").drop_duplicates(
            subset=["gaia_source_id"], keep="first"
        )

        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_final.to_csv(output_csv, index=False)

        print("\n" + "=" * 60)
        print(f"-> Crossmatch Pipeline Success! Clean entries: {len(df_final)}")
        print(f"-> Output saved at: {output_csv}")
        print("=" * 60)


if __name__ == "__main__":
    input_file = "data/raw/calspec_catalog.csv"
    output_file = "data/processed/crossmatch_catalog.csv"

    run_crossmatch(input_file, output_file, max_separation_arcsec=5.0, max_mag_diff=0.3)