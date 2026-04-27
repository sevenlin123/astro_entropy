import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# 1. Define a spectral model (Gaussian + continuum)
def spectral_model(wavelength, amp, center, sigma, cont_slope=0.0, cont_offset=1.0):
    """
    Simple linear continuum plus single Gaussian absorption line.
    Parameters:
        wavelength: array of wavelengths
        amp: amplitude of Gaussian (absorption depth)
        center: central wavelength of Gaussian
        sigma: width of Gaussian
        cont_slope: slope of linear continuum
        cont_offset: baseline offset
    Returns:
        flux array
    """
    continuum = cont_slope * wavelength + cont_offset
    gauss = amp * np.exp(-(wavelength - center) ** 2 / (2 * sigma ** 2))
    return continuum + gauss

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_spec, wav_min=4000, wav_max=7000, n_wave=1000):
    """Generate n_spec random spectra using the spectral_model."""
    wavelengths = np.linspace(wav_min, wav_min * 1?