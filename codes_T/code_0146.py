import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths):
    """Generate a simple multi‑Gaussian spectral model."""
    amps = [1.0, 0.8, 0.5]
    centers = [500, 600, 700]          # nm
    widths = [20, 30, 25]
    flux = np.zeros_like(wavelengths, dtype=float)
    for amp, cen, wid in zip(amps, centers, widths):
        flux += amp * np.exp(-((wavelengths - cen) ** 2) / (2 * wid**2))
    return flux

def generate_synthetic_spectra(n_spectra=10, n_points=300, rng=None):
    """
    Generate synthetic spectra from the base spectral model with random
    amplitude scaling and noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    wavelengths = np.linspace(400, 800, n_points)
    base = spectral_model(wavelengths)
    spectra = []
    for _ in range(n_spectra):
        scale = rng.uniform(0.5, 1.5)
        noise = rng.normal(scale=0.05, size=n_points)
        spectral_flux = (scale * base + noise)
        spectra.append((wavelengths, spectral_flux))
    return spectra

def photometric_filter(
    spectra,
    filters,
    wavelengths: any = None,
    **kwargs,
    ):
    """Simulate photometric measurements in multiple filters.
        This approach only based on content the … ….. ?????? 
    * ... ?