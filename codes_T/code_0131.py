import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV

# 1. Define a spectral model
def spectral_model(wavelengths, params):
    """Simple linear combination of Gaussian basis functions."""
    # params: [amplitudes..., mean1, std1, mean2, std2]
    n_basis = (len(params) - 4) // 2
    spectrum = np.zeros_like(wavelengths)
    for i in range(n_basis):
        amp = params[2 * i]
        mu  = params[2 * i + 1]
        sigma = params[2 * i + 2]
        spectrum += amp * np.exp(-0.5 * ((wavelengths - mu)/sigma)**2)
    return spectrum

# 2. generate synthetic spectra
def generate_synthetic_spectrum(wavelengths, true_params, noise_std=0.01):
    clean = spectral_model(wavelengths, true_params)
    noisy = clean + np.random.normal(scale=noise_std, size=len(wavelengths))
    return noisy

# 3. generate photometric data from synthetic spectra
def photometry_from_spectrum(spectrum, wavelengths, filters):
    """
    filters: list of tuples (center, width)
    returns integrated fluxes through each filter
    """
    fluxes = []
    for center, width in filters:
        mask = (wavelengths >= center - width/2) & (wavelengths <= center + width/2)
        flux = np.trapz(spectrum[mask], wavelengths[mask])
        fluxes.append(flux)
    return np.array(fluxes)

# 4. reconstruct a synthetic spectrum from photometric
def reconstruct_spectrum(photometry, wavelengths, filters, n_basis=3):
    """
    Fit linear combination of basis Gaussian kernels to match photometry.
    Use Ridge regression with cross-validation for regularization.
    """
    # build design matrix A
    basis_funcs = []
    for i in range(n_basis):
        mu = np.linspace(400, 700, n_basis)[i]
        sigma = 30
        basis = np.exp(-(wavelengths - mu)**s