import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

# 1) Define a simple linear spectral model: 
#    flux = A * basis + b
def build_spectral_basis(n_wave=200):
    """
    Create a set of orthogonal basis functions over a wavelength range.
    Returns array shape (n_wave, n_basis)
    """
    wave = np.linspace(0, 1, n_wave)
    # Two basis components: a flat and a linearly increasing component
    basis_flat = np.ones((n_wave, 1))
    basis_lin = wave.reshape(-1, 1)
    return np.hstack([basis_flat, basis_lin])

def spectral_model(coeffs, basis):
    """Compute synthetic spectrum from coefficients."""
    return coeffs @ basis.T

# 2) Generate synthetic spectra
def generate_synthetic_spectra(num_samples=100, n_wave=200):
    """Generate random synthetic spectra using the defined basis."""
    basis = build_spectrumi   # <-- This line...??   ???