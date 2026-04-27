import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ----------------------------------------------
# 1. Spectral model
# ----------------------------------------------
def create_spectral_model(wavelengths):
    """
    Create a simple linear spectral basis consisting of two Gaussian components.
    """
    mu = np.array([500., 700.])          # centers (nm)
    sigma = np.array([30., 40.])         # widths (nm)
    amplitude = np.array([1., 0.8])      # amplitudes

    basis = []
    for m, s, a in zip(mu, sigma, amplitude):
        component = a * np.exp(-0.5 * ((wavelengths - m) / s)**2)
        basis.append(component)
    return np.vstack(basis).T  # shape (n_wave, n_basis)

# ----------------------------------------------
# 2. Synthetic spectra generation
# ----------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths, rng=None):
    """
    Generate random linear combinations of the spectral basis.
    """
    rng = rng or np.random.default_rng()
    n_basis = 2
    coeffs = rng.normal(size=(n_samples, n_basis))
    basis = create_spectrum(wavelengths)
    return coeffs @ basis.T  # shape (n_samples, n_wave)

# --------------------------
# 3. Photometric data
# 
#   Photometric bandpasses (x-band, y=2x+1, etc.)
#   3..???    ;...
def calc_photometry(spectra, wavelengths, bandpasses):
    """
    **Note:** The use =!…   ...