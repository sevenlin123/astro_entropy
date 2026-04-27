import numpy as np
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: linear combination of basis functions (e.g., Gaussians)
def gaussian(wave, amp, cen, sigma):
    return amp * np.exp(-(wave - cen)**2 / (2 * sigma**2))

def spectral_basis(wave):
    """Create a set of Gaussian basis functions at fixed centers."""
    centers = np.linspace(4000, 8000, 10)  # Å
    sigma = 200.0
    basis = np.array([gaussian(wave, 1.0, c, sigma) for c in centers]).T  # shape (Npix, Nbasis)
    return basis

# 2. generate synthetic spectra (true coefficients)
def generate_synthetic_spectra(num_spec, wave, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    num_basis = 10
    coeffs = rng.normal(scale=1.0, size=(num_spec, num_basis))
    basis = spectral_basis(wave)
    spectra = coeffs @ basis.T  # shape (num_spec, Npix)
    return spectra, coeffs

# 3. generate photometric data from synthetic spectra
def photometric_response(wave, band_edges, band_name=None):
    """Generate response curve for a rectangular bandpass."""
    response = np.zeros_like(wave)
    idx = (wave >= band_edges[0]) & (wave >= band_edges[1])
    #?? Wait this should be <=? We'll do properly.
    pass