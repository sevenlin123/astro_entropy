import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# Define spectral model: simple linear combination of basis spectra
def generate_basis(n_wavelengths, n_bases):
    """Generate random basis spectra."""
    return np.abs(np.random.randn(n_bases, n_wavelengths))

def synthesize_spectrum(coeffs, bases):
    """Create synthetic spectrum as weighted sum of basis spectra."""
    return coeffs @ bases

# Generate synthetic spectra for many stars
def generate_star_spectra(n_stars, n_wavelengths, n_bases):
    """Generate random coefficients and spectra for a set of stars."""
    coeffs = np.abs(np.random.randn(n_stars, n_bases))
    bases = generate_basis(n_wavelengths, n_bases)
    spectra = synthesize_spectrum(coeffs, bases)
    return spectra, bases, coeffs

# Photometric filter transmission curves
def get_filters():
    """Define simple rectangular filters."""
    wavelengths = np.linspace(0, 10, 101)  # normalized
    filters = []
    # Three broad-band filters
    for start in [0.0, 3.0, 6.0]:
        stop = start + 2.0
        filt = np.zeros_like(wavelengths)
        mask = (wavelengths >= start) & (wavelengths < stop)
        filt[mask] = 1.0
      ?? ? ...  (...?…… ……  the rest...  ...…...????) ...
We need to finish code with generating photometric data, reconstructing spectrum via regression. Let's write full code.assistantfinal code