import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# ------------------ 1. Define a spectral model ------------------
def spectral_model(wavelengths, coeffs):
    """
    Simple linear combination of basis functions (e.g. Gaussians).
    
    Parameters:
    ----------
    wavelengths : ndarray
        Array of wavelengths (nm).
    coeffs : ndarray
        Coefficients for each basis function.
    
    Returns:
    -------
    spectrum : ndarray
        Synthetic flux values.
    """
    basis = np.exp(-(wavelengths[:, None] - 500 - 100 * np.arange(len(coeffs)))**2 / (2*50**2))
    return basis @ coeffs

# ------------------ 2. Generate synthetic spectra ------------------
def generate_synthetic_spectra(n_spec, n_wave, rng=None):
    """
    Generate random spectra using the spectral model.
    
      Parameters:
      ----------
      n_spec : int
          Number of spectra.
      n_wave : int
          Number filters or wavelength points.
      random: random generator for reproducibility.
      
    Returns:
    -------
    wavelengths : ndarray (Nw)
      **T**..?????...