import numpy as np
from scipy import interpolate
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------
# 1) Spectral model: simple linear combination of basis spectra
# -------------------------------------------------------------
def spectral_model(x, coeffs, basis):
    """
    Evaluate synthetic spectrum as a linear combination of basis spectra.
    
    Parameters
    ----------
    x : ndarray
        Wavelength grid.
    coeffs : ndarray
        Coefficients for each basis spectrum.
    basis : ndarray
        2D array (n_basis, n_wavelength) of basis spectra.
    
    Returns
    -------
    spectrum : ndarray
        Resulting synthetic spectrum.
    """
    return coeffs @ basis


# ------------------------------------------------------------------
# 2) Generate synthetic spectra (true coefficients & spectra)
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, n_wavelength, n_basis,
                              rng=np.random.default_rng()):
    """
    Generate random coefficients and corresponding spectra.
    
    Parameters
    ----------
    X : integer
        Number of samples.
    Y : integer 
        Weights to reflect 
    Y..??  # ???