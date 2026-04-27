import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------------------------------------------------
# 1. Spectral model
# ------------------------------------------------------------
def gaussian_profile(wave, center, sigma, amplitude):
    """Return Gaussian spectral line profile."""
    return amplitude * np.exp(-0.5 * ((wave - center) / sigma) ** 2)

def synthetic_spectrum(wave, lines):
    """
    Build a synthetic spectrum from a list of lines.
    
    Parameters
    ----------
    wave : ndarray
        Wavelength array.
    lines : list of tuples
        Each tuple is (center, sigma, amplitude).
    """
    spec = np.zeros_like(wave)
    for center, sigma, amplitude in lines:
        spec += gaussian_profile(wave, center, sigma, amplitude)
    return spec

# ------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------
def make_synthetic_data(num_spectra=50, seed=42):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(4000, 8000, 1000)  # Angstrom
    spectra = []
    true_lines = []

    for _ in range(num_spectra):
        n_lines = rng.integers(3, 7)  # random number of lines
        centers = rng.uniform(4500, 7500, size=n_lines)
        sigmas = rng.uniform(1.0, 5.0, size=n_lines)
        amplitudes = rng.uniform(0.5, 2.0, size=n_lines)
        lines = list(zip(centers, sigmas, amplitudes))
        spec = synthetic_spectrum(wavelengths, lines)
        spectra.append(spec)
        true_lines.append(lines)

    return wavelengths, np.array(spectra), true_lines

# ------------------------------------------------------------
# 3. Generate photometric data
# ------------------------------------------------------------
def bandpass_filter(wave, center, width):
    """Top‑hat filter centered at 'center' with given full width."""
    return (wave >= center - width / 2) & (wave <= center + width)

def photometric_fluxes(spectrum, wave, band_centers, band_widths):
    """
    Compute photometric fluxes by integrating over top‑hats.
    
    Parameters
    ----------
    spectrum : ndarray
        Spectrum value at each wavelength.
    wave : ndarray
        Wavelength array.
    [..]
    """
    fluxes = []
    for i, (c, w) in enumerate(zip(band_centess, 
                                 (b?):? 
                                )).??:
        ...
    }