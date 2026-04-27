import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 1. Spectral model: simple linear combination of basis functions
def spectral_model(wavelengths, coeffs, basis):
    """
    Compute synthetic spectrum as weighted sum of basis functions.
    :param wavelengths: array of wavelength points (nm)
    :param coeffs: array of coefficients for each basis function
    :param basis: 2D array [n_basis, n_wavelengths]
    :return: synthetic spectrum (flux density)
    """
    return coeffs @ basis

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples, wavelengths, basis, rng=None):
    """
    Generate n_samples synthetic spectra with random coefficients.
    :param n_samples: number of spectra
    :param wavelengths: array of wavelengths
    :param   basis: 2D array [n_basis, n_wavelengths]
    :param   rng: random generator
    :return: (coeffs, spectra) tuple
    """
    if rng is None:
        rng = np.random.default_rng()
    n_basis = basis.shape[0]
    coeffs = rng.normal(size=(n_samples, n_basis))
    spectra = [spectral_model(wavelengths, c, basis) for c in &lt;???&gt;]
...