"""
Module: dataset_utils.py

This module provides functions to load CALSPEC FITS spectra, convert them into a
consistent physical units grid, degrade spectral resolution to resemble Gaia
BP/RP observations, and simulate Gaia-like instrumental noise for model training.
"""

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

#def load_calspec_fits(fits_path: str)

#def degrade_spectral_resolution()