import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ---------------------------------------------
# 1. Define a simple spectral model
# ---------------------------------------------

def generate_spectrum(wavelengths, coeffs):
    """
    Simple linear combination of Gaussian basis functions.
    
    Parameters:
        wavelengths : array-like, shape (N,)
            Wavelength grid in nm.
        coeffs     : array-like, shape (M,)
            Coefficients for each Gaussian basis function.
    
    Returns:
        flux : array-like, shape (N,)
            Generated flux spectrum.
    """
    gauss_centers = np.linspace(400, 800, len(coeffs))
    gauss_sigma   = 20.0
    
    flux = np.zeros_like(wavelengths, dtype=float)
    for a, mu in zip(coeffs, gauss_centers):
        flux += a * np.exp(-((wavelengths - mu) ** 2) / (2 * gauss_sigma ** 2))
    return flux

# ---------------------------------------------
# 2. Generate synthetic spectra
# ---------------------------------------------

def create_synthetic_dataset(num_samples=200, num_wavelengths=500):
    """
    Generate a dataset of synthetic spectra and random coefficients.
    
    Parameters:
        num_samples      : int
            Number of spectra to generate.
        num_wavelengths : int
            Number of wavelength points in each spectrum.
    
        return:
        spectra   : array shape (num_samples, num_wavelengths)
        coeffs    : array shape (num_samples, M)
        wavelengths : ndarray
            Common wavelength grid.
    """
    wavelengths = np.linspace(400, 800, num_wavelengths)
    M = 5  # number of Gaussian basis functions
    coeffs = np.random.normal(size=(num_samples, M))
    spectra = np.array([generate_spectrum(wavelengths, c) for c in coeffs])
    # Add some random noise
    noise = np.random.normal(scale=0.02, size=spectra.shape)
    spectra = spectra + noise * abs(Spectral???...
...