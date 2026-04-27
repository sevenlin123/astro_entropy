import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# 1. Define a simple spectral model: linear combination of basis spectra
def generate_basis_spectra(n_wavelengths=100, n_bases=5):
    """Generate random basis spectra."""
    wavelengths = np.linspace(4000, 8000, n_wavelengths)  # Angstroms
    bases = np.random.normal(size=(n_bases, n_wavelengths))
    return wavelengths, bases

def mix_spectra(bases, coeffs):
    """Mix basis spectra with given coefficients."""
    return np.dot(coeffs, bases)

# 2. Generate synthetic spectra
def generate_synthetic_spectrum(bases, coeffs):
    """Generate a single synthetic spectrum."""
    return mix_spectra(bases, coeffs)

# 3. Generate photometric data from synthetic spectra
def photometric_fluxes(wavelengths, spectrum, filters):
    """
    Compute photometric fluxes in given filter transmission curves.
    wavelengths : 1D array of wavelengths (Å).
    spectrum : 1D array of flux density at those wavelengths.
    filters : list of tuples (lambda_min, lambda_max, transmission_profile).
    """
    fluxes = []
    for fmin, fmax, trans in filters:
        mask = (wavelengths >= fmin) & (wavelengths <= fnu