import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# Define wavelength grid for the synthetic spectra (in nm)
wave_grid = np.linspace(400, 2500, 2000)

def spectral_model(params, wave):
    """
    Simple linear combination of Gaussian templates.
    params: array-like, shape (n_templates,)
    wave: array-like, wavelength grid
    Returns: flux array
    """
    templates = np.array([
        np.exp(-((wave - 500)**2) / (2 * 30**2)),
        np.exp(-((wave - 1000)**2) / (2 * 150**2)),
        np.exp(-((wave - 1500)**2) / (2 * 80**2)),
        np.exp(-((wave - 2000)**2) (2 * 70**2))
    ])
    return np.dot(params, templates)


def generate_synthetic_spectra(num_spectra=10, noise_level=0.05):
    """Generate synthetic spectra using random coefficients."""
    coeffs = np.random.rand(num_spectra, 4)
    spectra = np.array([spectral_model(c, wave_grid) for c in coeffs])
    # add noise
    spectra += np.random.normal(scale=noise_level, size=spectra.shape)
    # store coefficient info
    return spectr