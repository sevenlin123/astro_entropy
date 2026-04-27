import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple linear spectral model
def create_spectral_model(wavelengths, n_basis=5):
    """
    Create a set of basis functions over the wavelength grid.
    Each basis is a Gaussian centered at random positions.
    """
    rng = np.random.default_rng()
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), n_basis)
    widths = rng.uniform(10, 30, n_basis)
    basis = np.array([np.exp(-0.5 * ((wavelengths - c) / w) ** 2)
                      for c, w in zip(centers, widths)])
    return basis.T  # shape (n_pixels, n_basis)

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples, wavelengths, basis, noise_std=0.01):
    """
    Generate synthetic spectra as linear combos of the basis.
    """
    rng = np.random.default_rng()
    coeffs = rng.normal(size=(n_samples, basis.shape[1]))
    raw = coeffs @ basis.T  # (n_samples, n_pixels)
    return raw + np.random.normal(scale=noise_std, size=raw.shape)

# 3. **RETRIEVE** : get them from **..??**
def _get_floater?