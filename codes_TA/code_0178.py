import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso

# ----------------------------------------------------------------------
# 1) Define a simple spectral model
# ----------------------------------------------------------------------
def spectral_model(wave, params):
    """Generate a Gaussian spectral line profile.

    Parameters
    ----------
    wave : ndarray
        Wavelength grid (in nm).
    params : tuple
        (amplitude, center, width). Each is a scalar.

    Returns
    -------
    flux : ndarray
        Flux values at given wavelengths.
    """
    amp, cen, wid = params
    return amp * np.exp(-((wave - cen)**2) / (2 * wid**2))

# ----------------------------------------------------------------------
# 2) Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_specs, wave_grid):
    """Produce n_specs random spectra by sampling spectral parameters.

    Parameters
    ----------
    n_specs : int
        Number of spectroscopic observations.
    wave_grid : ndarray
        Wavelength grid for the spectra.