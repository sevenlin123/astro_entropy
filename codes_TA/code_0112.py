import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple linear spectral model
def generate_spectral_model(n_wavelengths=1000, seed=42):
    """Generate a synthetic stellar spectral template.
    Parameters:
        n_wavelengths: Number of wavelength points.
    Returns:
        wavelengths (np.ndarray): Wavelength grid in nm.
        template (np.ndarray): Template spectrum (arbitrary units).
    """
    np.random.seed(seed)
    # Create a smooth continuum
    base = np.linspace(1.0, 0.8, n_wavelengths)
    # Add random Gaussian absorption lines
    for _ in range(5):
        center = np.random.uniform(400, 800)
        width = np.random.uniform(5, 20)
        amp = np.random.uniform(0.05, 0.15)
        line = np.exp(-0.5 * ((np.arange(n_wavelengths) - center)**2) / width**2)
        line = 1.0 - amp * line
        base += line - 1.0
    wavelengths = np.linspace(350, 1050, n_wavelengths)
    return wavelengths, base

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_spectra=10, n_wavelengths=1000, seed=123):
    """Generate multiple synthetic spectra by scaling and adding noise."""
    wavelengths, template = generate_spectral_model(n_wavelengths, seed)
    spectra = []
    for i in range(n_spectra):
        scale = 0.8 + 0.4 * np.random.rand()
        noise = np.random.normal(0, 0.01, n_wavelengths)
        spect = scaling(scalef, template, noise)  # <-- Problem
...