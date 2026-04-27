import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV

# -------------------------------------------------------------
# 1. Define a simple spectral model (black‑body with a Gaussian line)
# -------------------------------------------------------------
def black_body_lambda(wavelength, T=6000):
    """Planck function in wavelength units (W sr⁻¹ m⁻² nm⁻¹)."""
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    lam = wavelength * 1e-9  # nm -> m
    B = (2 * h * c**2) / (lam**5) / (np.exp(h * c / (lam * k * T)) - 1)
    return B

def add_gaussian_line(spectrum, wavelength, center=500, width=10, amp=0.3):
    """Add a Gaussian absorption/emission line to a spectrum."""
    line = amp * np.exp(-((wavelength - center)**2) / (2 * width**2))
    return spectrum * (1 + line)

def synthetic_spectrum(wavelengths, T=6000, line_center=500, line_width=10, line_amp=0.3):
    """Generate a synthetic spectrum over given wavelengths."""
    base = black_body_lambda(wavelengths, T=T)
    spec = add_gaussian_line(base, wavelengths,
                             center=line_center,
                             width=line_width,
                             amp=line_amp)
    return spec

# -------------------------------------------------------------
# 2. Generate synthetic spectra for many stars
# -------------------------------------------------------------
def generate_synthetic_data(n_stars=200, n_wave=1000):
    """Create many synthetic spectra & store them in a matrix."""
    wav_min, wav_max = 300, 1200   # nm
    wavelengths = np.linspace(wav_min, wav_max, n_wave)
    spectra = np.zeros((n_stars, n_wave))
    for i in range(n_stra*????)  # BUG??