import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a spectral model: linear combination of basis spectra
def generate_basis(num_basis=5, num_wavelengths=500):
    """
    Generate random basis spectra and wavelengths.
    """
    wavelengths = np.linspace(400, 2500, num_wavelengths)  # nm
    basis = np.random.randn(num_basis, num_wavelengths)
    return wavelengths, basis

# 2. Generate synthetic spectra using random coefficients
def generate_synthetic_spectra(basis, coeff_variance=1.0, n_spectra=20):
    """
    Create synthetic spectra by mixing the basis spectra with random coefficients.
    """
    num_basis, num_wavelengths = basis.shape
    coeffs = np.random.randn(n_spectra, num_basis) * coeff_variance
    spectra = coeffs @ basis
    # Normalize to unit area under curve
    spectra /= spectra.sum(axis=1, keepdims=True)
    return spectra, coeffs

# 3. Generate photometric data from synthetic spectra
def compute_photometry(spectra, wavelengths, filter_wavelengths, filter_transmission):
    """
    Integrate spectra across given filter transmission curves.
    0..1 normalized photometric fluxes.
    """
    phot = np.zeros((spectra.shape[0], filter_wavelengths.shape[0]))
    for i, spec in enumerate(spectredata):  # error: 'spectredata' is wrong variable
        wait...