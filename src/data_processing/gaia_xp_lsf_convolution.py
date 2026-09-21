"""
Module: spectral_degradation.py
Report Section: 4.4 Spectral Resolution Degradation to Gaia XP DR3

This module provides routines to degrade high-resolution reference spectra (e.g., CALSPEC)
to match the instrument line spread function (LSF) and variable spectral resolution (R) 
of Gaia DR3 externally calibrated XP spectra (Montegriffo et al. 2023).
"""

import numpy as np
from numpy.typing import NDArray

def degrade_to_gaia_xp(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    xx: NDArray[np.float64],
) -> NDArray[np.float64]:
   """
    Degrade high-resolution spectra to the variable resolution of Gaia DR3 XP spectra.

    Convolves input spectral flux densities with a wavelength-dependent Gaussian 
    kernel whose Full Width at Half Maximum (FWHM) follows the Gaia DR3 externally 
    calibrated spectra (ECS) resolution curve R(lambda). Resamples the degraded 
    flux onto a target wavelength grid.

    Parameters
    ----------
    x : numpy.ndarray
        Input wavelength array for reference spectra (e.g., in nanometers or Angstroms).
    y : numpy.ndarray
        Input flux density array corresponding to `x`. Can be 1D (single spectrum, 
        shape `(N,)`) or 2D (multiple spectra, shape `(M, N)` or `(N, M)`).
    xx : numpy.ndarray
        Target wavelength grid array for output resampled spectra.

    Returns
    -------
    ys : numpy.ndarray
        Degraded and resampled flux density array evaluated at wavelength grid `xx`.

    Notes
    -----
    - Resolution table values R = lambda / FWHM are based on Montegriffo et al. (2023) 
      Gaia DR3 XP calibration data.
    - FWHM is converted to Gaussian standard deviation via:
      sigma = FWHM / (2 * sqrt(2 * ln(2)))
    """
   # Table describing the resolution R = lambda / FWHM of Gaia DR3 XP spectra (Montegriffo et al. 2023)
   xt = np.array([350.,370.,390.,410,430.,450.,470.,490.,510.,530.,550.,570.,590.,610.,630.,640.,660.,680.,700.,720.,740.,760.,780.,800.,820.,840.,860.,880.,900.,920.,940.,960.,980.])
   yt = np.array([71.5,67.2,61.4,55.3,49.9,45.0,41.4,38.0,35.0,32.5,30.0,28.3,25.5,25.2,22.0,74.6,77.8,77.8,76.8,74.5,73.2,71.1,68.8,66.9,65.1,63.4,62.1,60.7,58.2,57.2,54.9,53.8,52.0])
   yt = xt/yt # FWHM = lambda/R

   # If working with nm, convert to Angstroms for the convolution
   #xt = xt*10.
   #yt = yt*10.

   gg = np.zeros((len(x),len(x)))

   for i in range(len(x)):
      sigma=np.interp(x[i],xt,yt)/2.0/np.sqrt(-2.0*np.log(0.5))
      kernel = np.exp(-(x-x[i])**2/2./sigma**2)
      gg[i,:] = kernel/np.sum(kernel)

   rr = np.matmul(gg,np.transpose(y))
   d2 = np.transpose(rr)
   ys = np.interp(xx,x,d2)

   return(ys)