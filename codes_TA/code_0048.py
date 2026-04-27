import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# Define spectral model: Gaussian basis functions
def gaussian_basis(wavelengths, centers, widths):
    """
    Generate a set of Gaussian basis functions evaluated at wavelengths.
    """
    return np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :])**2)

# Generate synthetic spectra
def generate_synthetic_spectra(n_spectra, wavelengths, n_bases, noise_level=0.01):
    """
    Create n_spectra synthetic spectra using random linear combinations of Gaussian bases.
    """
    rng = np.random.default_rng()
    centers = np.linspace(400, 800, n_bases)
    widths = rng.uniform(10, 30, size=n_bases)
    basis = gaussian_basis(wavelengths, centers, widths)   # shape (len(wavelengths), n_bases)
    coeffs = rng.normal(size=(n_spectra, n_bases))
    spectra = basis @ coeffs.T                                   # (len(wavelengths), n_spectrum)
    noise = rng.normal(scale=noise_level, size=spectra.shape)
    return spectra.T, basis, coeffs, widths

# Photometric filter definitions (x,y luminance vs.
def copy and ..????