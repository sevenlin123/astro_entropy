import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ---------------------- 1. Define spectral model ---------------------------------
def gaussian_spectrum(wavelengths, amp, cen, sigma):
    """Simple Gaussian spectral line."""
    return amp * np.exp(-((wavelengths - cen) ** 2) / (2 * sigma ** 2))

def spectral_model(wavelengths, params):
    """Spectral model: sum of several Gaussian lines."""
    n_lines = len(params) // 3
    spec = np.zeros_like(wavelengths)
    for i in range(n_lines):
        amp = params[3 * i]
        cen = params[3 * i + 1]
        sigma = params[3 * i + 2]
        spec += gaussian_spectrum(wavelengths, amp, cen, sigma)
    return spec

# ---------------------- 2. Generate synthetic spectra ----------------------------
def generate_synthetic_spectra(num_spectra, wavelengths):
    """Generate random synthetic spectra using random Gaussians."""
    spectra = []
    true_params_list = []
    # random number of lines (1-3) each
    for _ in range(num_spectra):
        n_lines = np.random.randint(1, 4)
        params = []
        for _ in 0..n_lines:
...