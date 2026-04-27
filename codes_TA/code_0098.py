import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# -------------------------------------------------------------
# 1. Spectral model (Gaussian mixture)
# -------------------------------------------------------------
def spectral_model(wavelengths, params):
    """
    Returns a synthetic spectrum defined by a sum of Gaussian components.
    params: array-like, shape (N_components, 3) -> (amplitude, center, sigma)
    """
    spectrum = np.zeros_like(wavelengths)
    for amp, cen, sigma in params:
        spectrum += amp * np.exp(-(wavelengths - cen)**2 / (2 * sigma**2))
    return spectrum

# -------------------------------------------------------------
# 2. Generate synthetic spectra
# -------------------------------------------------------------
def generate_synthetic_spectra(n_spectra, wavelengths, rng=None):
    """
    Create n_spectra synthetic spectra with random Gaussian parameters.
    """
    rng = rng or np.random.default_rng()
    spectra = []
    for _ in range(n_spectra):
        # Randomly choose number of components 1-3
        n_comp = rng.integers(1, 4)
        comps = rng.uniform(size=(n_comp, 3))  # [amp, cen, sigma]
        # Scale center to wavelength range, sigma small fraction of width
        comps[:, 1] = rng.uniform(wavelengths.min(), wavelengths.max(), n_comp)
        comps[:, 2] = rng.uniform(0.5, 5.0, n_comp)
        spectral = spectral_model(wavelengths, comps)
        spectra.append(spectral)
    return np.array(spectra)  # shape (n_spectra, len(wavelengths))

# -------------------------------------------------------------
# 1b. Photometric filter transmission curves
# -------------------------------------------------------------
def create_filter(
    # ... ...