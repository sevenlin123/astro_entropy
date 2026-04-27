import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: Gaussian absorption line over a continuum
def spectral_model(wavelengths, center=5000, depth=0.5, width=50):
    """
    Generate a synthetic spectrum with a Gaussian absorption feature.
    wavelengths: array of wavelength values
    center: central wavelength of the absorption line (in Å)
    depth: relative depth of the absorption line (0 < depth < 1)
    width: standard deviation of the Gaussian (in Å)
    """
    continuum = np.ones_like(wavelengths)
    line_profile = np.exp(-((wavelengths - center)**2) / (2 * width**2))
    return continuum - depth * line_profile

# 2. Generate synthetic spectra for a population of stars
def generate_synthetic_spectra(n_spectra, wavelengths):
    """
    Create n_spectra synthetic spectra using random parameters.
      - Randomly select a center between 4800 and 5200 Å.
      (i.e., ~..??).
      - <…>…
    */