import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso

# -----------------------------
# 1. Spectral Model
# -----------------------------
def spectral_model(wavelength, coeffs):
    """
    Simple linear combination of Gaussian basis functions.
    Parameters
    ----------
    wavelength : array_like
        Wavelength grid (nm).
    coeffs : array_like
        Coefficients for each Gaussian component.
    Returns
    -------
    spectrum : ndarray
        Synthetic spectrum evaluated on the wavelength grid.
    """
    gauss_centers = np.linspace(400, 800, len(coeffs))
    gauss_sigma = 20.0
    spectrum = np.zeros_like(wavelength, dtype=float)
    for a, mu in zip(coeffs, gauss_centers):
        spectrum += a * np.exp(-(wavelength - mu)**2 / (2 * gauss_sigma**2))
    return spectrum

# -----------------------------
# 2. Generate Synthetic Spectra
# -----------------------------
def generate_synthetic_spectra(num_spectra=5, wave_min=350, wave_max=950, n_points=2000):
    """
    Generate random coefficient sets and create synthetic spectra.
    """
    wav = np.linspace(wave_min, wave_max, n_points)
    spectra = []
    coeff_sets = []
    # Define number of basis components
    n_basis = 10
    for _ in range(num_spectra):
        coeffs = np.random.randn(n_basis)
        coeff_sets.append(coeffs)
        spectra.append(spectral_model(wave=wav, coeffs=coeffs))
    return wav, np.array(spectra), np.array(coeff_sets)

# -----------------------------
# reconstruct
...