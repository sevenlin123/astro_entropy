import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def spectral_model(wavelengths):
    """
    Simple spectral model: mixture of Gaussian absorption lines on a continuum.
    Parameters:
        wavelengths : array-like, shape (n_points,)
    Returns:
        fluxes : array-like, shape (n_points,)
    """
    # Continuum
    continuum = 1.0 + 0.1 * np.sin(2 * np.pi * wavelengths / 500.0)
    
    # Gaussian absorption features
    gaussians = np.zeros_like(wavelengths)
    line_centers = [550, 600, 650]
    line_sigma = [5.0, 8.0, 10.0]
    for center, sigma in zip(line_centers, line_sigma):
        gaussians += -0.3 * np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    
    return continuum + gaussians

def generate_synthetic_spectra(num_spectra=100, num_points=200):
    """
    Generate a set of synthetic spectra using the spectral model.
      - Each spectrum is perturbed by random scaling and noise.
    1. Create a wavelength grid.
    2. Compute base model.
     Flux = Base * (1 + perturbation) + noise.
    """
    wavelengths = np.linspace(400, 800, num_points)
    base_flux = spectral_model(wavelengths)
    spectra = []
    for _ in range(num_spectrum):
        scale = 1.0 + np.random.normal(0, 0.05)
        noise = np.random.normal(0, 0.01, size=num_points)
        spectral = base_flux * scale + noise