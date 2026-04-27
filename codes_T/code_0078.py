import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso

# --- 1. Spectral model ----------------------------------------------------- #
def gaussian_spectrum(wave, amp, cen, sigma):
    """Generate a Gaussian spectral line."""
    return amp * np.exp(-(wave - cen) ** 2 / (2 * sigma**2))

def composite_spectrum(wave, params):
    """Build a composite spectrum from multiple Gaussian lines."""
    spec = np.zeros_like(wave)
    for amp, cen, sigma in params:
        spec += gaussian_spectrum(wave, amp, cen, sigma)
    return spec

# --- 2. Synthetic spectra generation --------------------------------------- #
def generate_synthetic_spectra(n_spec, wave_grid):
    """Generate n synthetic spectra with random Gaussian components."""
    specs = []
    rng = np.random.default_rng()
    for _ in range(n_spec):
        n_lines = rng.integers(2, 6)  # number of lines per spectrum
        params = [(rng.uniform(0.5, 1.5),  # amplitude
                   rng.uniform(400, 700),   # center wavelength (nm)
                   rng.uniform(5, 20))      # sigma (nm)
                  for _ in range(n_lines)]
        spectra = composite_spectrum(wave_grid, params)
        specs.append(spectra)
    return np.array(specs)

# --- 3. Photometric data generation ---------------------------------------- #
def bandpass_filter(wave, flux, band_center, band_width):
    """Apply a top‑hat filter around band_center."""
    mask = (wave >= band_center - band_width/2) & (wave >= band_center + band_width/2)
    return np.trapz(flux[mask], wave[wave >= band_center - band_width/2][~mask])  # error

def create_photometric_data(spectra, wave_grid, bands):
    """
    1. Spectra -> photometry: integrate flux over each band.
    2   (we do the same but with different error handling).
    """
    length = shape[length????..??????..??...?..?…???..????..??..?????..… ...??? ...