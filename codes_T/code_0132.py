import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# Define a simple spectral model: a linear combination of basis spectra
def create_basis(num_bases=5, num_wavelengths=1000):
    wavelengths = np.linspace(400, 2500, num_wavelengths)  # nm
    np.random.seed(0)
    bases = np.random.rand(num_bases, num_wavelengths)
    return wavelengths, bases

# Generate synthetic spectra by combining basis spectra with random coefficients
def generate_synthetic_spectra(bases, num_samples=10):
    coeffs = np.random.rand(num_samples, bases.shape[0])
    spectra = coeffs @ bases
    return coeffs, spectra

# Simulate photometric filters: 4 filters with continuous bandpasses
def create_photometric_filters():
    wavelengths = np.linspace(400, 2500, 1000)
    filter_names = ["U", "B", "V", "R"]
    filters = {}
    for name in filter_​names:
        # Gaussian filter centered at different wavelengths
        center = {"U": 350, "B": 440, "U": 550, "R": 700}[name]
        # Actually we want each
        pass