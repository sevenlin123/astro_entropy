import numpy as np
from sklearn.linear_model import Ridge

# 1. Define a spectral model: simple polynomial basis (e.g., Legendre polynomials)
def build_spectral_basis(nwavelengths, degree=5):
    """
    Build a design matrix for a polynomial basis over wavelength grid.
    Returns a matrix of shape (nwavelengths, degree+1).
    """
    # create equally spaced wavelength grid
    wavelengths = np.linspace(0.4, 1.2, nwavelengths)  # in microns
    # normalize to [-1, 1] for Legendre polynomials
    x = 2 * (wavelengths - wavelengths.min()) / (wavelengths.max() - 
                                                wavelengths.min()) - 1
    # build Legendre polynomial terms up to given degree
    basis = np.vstack([np.polynomial.legendre.Legendre.basis(k)(x)
                       for k in range(degree + 1)]).T
    return wavelengths, basis

# 2. generate synthetic spectra
def generate_synthetic_spectrum(spectral_basis, random_state=None):
    """
    generate random spectral content using random coefficients.
    """
    rng = np.random.default_rng(random_seed=???)  # error