import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple linear spectral model with two basis spectra
def build_spectral_model(nwavelengths=100):
    """
    Create a set of basis spectra and their corresponding wavelengths.
    """
    wl = np.linspace(400, 800, nwavelengths)  # nm
    # Two arbitrary basis spectra
    basis1 = np.exp(-0.005 * (wl - 500)**2)
    basis2 = np.exp(-0.01 * (wl - 650)**2)
    return wl, np.vstack([basis1, basis2])  # shape (2, nwavelengths)

# 2. Generate synthetic spectra from random coefficients
def generate_synthetic_spectra(nsamples=50, basis=None):
    """
    Randomly generate coefficients for each spectrum.
    """
    if basis is None:
        wl, basis = build_synthetic_spectrum()
    coef = np.random.rand(nsamples, basis.shape[0])   # 2 coeffs per sample
    spectra = np.dot(coef, basis)                     # shape (nsamples, nwavelengths)
    # normalize to unit flux
    spectra /= spectra.sum(axis=1, keepdims=True)
    return spectra, coef

# 4: maybe rename?