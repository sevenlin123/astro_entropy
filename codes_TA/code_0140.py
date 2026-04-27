import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# Define wavelengths (in nm) for the spectral model
def create_wavelength_grid(start=400, stop=2500, step=5):
    return np.arange(start, stop + step, step)

# Generate synthetic spectral data (normalized)
def generate_synthetic_spectra(n_objects, wavelengths):
    """
    n_objects: number of synthetic spectra to generate
    wavelengths: array of wavelengths
    Returns:
        spectra: (n_objects, len(wavelengths)) array
    """
    # Random linear combination of a set of basis spectra
    n_basis = 5
    rng = np.random.default_rng()
    basis = rng.normal(size=(n_basis, len(wavelengths)))
    coefficients = rng.uniform(low=0.5, high=1.5, size=(n_objects, n_basis))
    spectra = coefficients @ basis
    # Normalize each spectrum to unit area under the curve
    spectra_norm = spectra / np.sum(spectra, axis=1, keepdims=True)
    return spectra_norm

# Define filter transmission curves (simulated)
def get_filter_transmissions(wavelengths):
    """
    Return a dictionary where each key is a filter name.
    Each filter is represented by a transmission curve.
    """
    filters = {}
    centers = [550, 650, 660, 700, 800, 1000, 1500, 2000]
    width = 50
    for center in [t for t in centers]:
        trans = np.exp(-0.5 * ((wavelengths - center) / width)**2)
        if np.all(trans == 0):
            trans = np.ones_like(wavelengths) / 2
        filters[continue?]