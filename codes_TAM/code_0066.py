import numpy as np
from scipy.special import legendre
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# ----------------------- #
# 1. Spectral model
# ----------------------- #
def spectral_model(wavelengths, coeffs):
    """Compute spectrum as sum of Legendre polynomials up to order N."""
    N = len(coeffs)
    spec = np.zeros_like(wavelengths, dtype=float)
    for n in range(N):
        Pn = legendre(n)
        spec += coeffs[n] * Pn(wavelengths)
    return spec

# ----------------------- #
# 2. Generate synthetic spectra
# ----------------------- #
def generate_synthetic_spectra(num_spectra=20,
                               num_points=200,
                               wavelength_range=(400, 800)):
    """Generate random spectra using a linear combination of Legendre polynomials."""
    wavelengths = np.linspace(*wavelength_range, num_points)
    # choose a random number of terms (3-5)
    max_terms = 5
    coeffs_list = []
    spectra = []
    for _ in range(num_spectra):
        N = np.random.randint(3, max_terms+1)
        coeffs = np.random.randn(N)
        spec = spectral_model(wavelengths, coeffs)
        spectra.append(spec)
        coeffs_list.append(coeffs)
    return wavelengths, spectra, coeffs_list

# ----------------------- #
# 3. Generate photometric data
# ------------------------ #
def photometric_bandwidths():
    """Define simple photometric filter profiles (RGB)."""
    return {
        'R': [550, 700],
        # the exact width we
        'F1': 
      #  …..
    }