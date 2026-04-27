import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# -----------------------------
# 1. Define a spectral model
# -----------------------------
def spectral_model(wavelengths, coeffs):
    """
    Simple linear combination of basis spectra.
    """
    # Two basis spectra: Gaussian + linear background
    gauss = np.exp(-0.5 * ((wavelengths - 500) / 50)**2)
    linear = 1 + 0.01 * (wavelengths - 400)
    return coeffs[0] * gauss + coeffs[1] * linear

# ---------------------------------
# 2. Generate synthetic spectra
# ---------------------------------
def generate_synthetic_spectra(n_samples=100, wavelengths=np.arange(400, 701)):
    """
    Generate n_samples spectra using random coefficients.
    """
    spectra = []
    for _ in range(n_samples):
        coeffs = np.random.uniform(0.5, 1.5, size=2)
        spectra.append(spectral_model(wavelengths, coeffs))
    return np.array(spectra), coeffs

# ------------------------------------
# 3. Generate photometric data
# ------------------------------------
def bandpass_filter(spectrum, wavelengths, center, width=20):
    """
    calculate photometric flux for a band with Gaussian filter
    :param:   (x, y)
        start point
        end … ...
   ..???????????