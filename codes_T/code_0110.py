import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# 1. Define a simple spectral model: linear combination of basis spectra
def basis_spectra(wavelengths, n_basis=5):
    """
    Generate synthetic basis spectra (smooth curves).
    """
    np.random.seed(0)
    basis = []
    for i in range(n_basis):
        amp = np.random.uniform(0.5, 1.5)
        shift = np.random.uniform(-50, 50)
        width = np.random.uniform(100, 300)
        spec = amp * np.exp(-((wavelengths - shift) ** 2) / (2 * width ** 2))
        basis.append(spec)
    return np.vstack(basis)  # shape (n_basis, N)

# 2. Generate synthetic spectra by mixing basis spectra
def generate_synthetic_spectra(n_samples, wavelengths):
    """
    Create spectra with random weights for each basis component.
    """
    n_basis = 5
    basis = basis_spectra(wavelengths, n_basis)
    coeffs = np.random.rand(n_samples, n_basis)
    spectra = coeffs @ basis  # shape (n_samples, N)
    return spectra, coeffs

# 3. Simulate photometric data: integrate over filter transmission curves
def simulate_photometry(spectra, wavelengths, filters):
    """
    filters: list of tuples (center, width)
      returns integrated fluxes per filter
    """
    phots = []
    for (c, w) in filters:
        mask = (wavelengths >= c - w/2) & (wavelengths <= c + w/2)
        trans = np.ones_like(wavelengths)
        trans[~mask] = 0
        flux = spectra @ trans  # shape (n_samples,)
        flux -= flux.min()  # zero‑line adjustment
        flux1 = self? ????
?