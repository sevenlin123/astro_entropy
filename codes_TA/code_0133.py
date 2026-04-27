import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV

# 1. Spectral model: simple linear combination of basis spectra
def generate_basis(num_bases=3, wave_start=400, wave_end=800, num_points=201):
    """Generate random basis spectra over wavelength grid."""
    wavelengths = np.linspace(wave_start, wave_end, num_points)
    # Randomly generate basis spectra with positive flux
    bases = np.random.rand(num_bases, num_points) + 0.1
    return wavelengths, bases

def synth_spectrum(basis, coeffs):
    """Linear combination of basis spectra with given coefficients."""
    return np.tensordot(coeffs, basis, axes=([0], [0]))

# 2.  generate synthetic spectra
def create_synthetic_spectra(num_spectra=5, basis=None, noise_std=0.05):
    """
    generate two random coefficients per spectrum.
    """
    if basis is not None:
        _, num_basis = basis.shape
    else:
        # default basis
        wavelengths, basis = generate_basis()
        waverange = [wavelengths[0] ,..???,.. ???