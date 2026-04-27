import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: linear combination of basis spectra
def generate_basis(num_bases=5, num_points=200):
    """Generate random basis spectra."""
    wavelengths = np.linspace(400, 1000, num_points)  # nm
    basis = np.random.rand(num_bases, num_points)
    return wavelengths, basis

def synthesize_spectrum(basis, coeffs):
    """Create a synthetic spectrum from coefficients."""
    return np.dot(coeffs, basis)

# 2. Generate synthetic spectra
def create_synthetic_dataset(num_spectra=50, basis=None, rng=np.random.default_rng()):
    if basis is None:
        wavelengths, basis = generate_basis()
    else:
        wavelengths, _ = generate_basis()  # just to get wavelengths
    coeffs = rng.normal(size=(num_spectra, basis.shape[0]))
    spectra = np.array([synthesize_spectrum(basis, c) for c in coeffs])
    return wavelengths, spectra, coeffs

# 3. photometric data from synthetic spectra
def apply_filter(wavelengths, spectrum, filter_center, width):
    """
    Simple top‑hue filter.
    0..1 range.
    **We can't do full convolution.**  \
    **For simplicity, assume we only focus on the center region.**