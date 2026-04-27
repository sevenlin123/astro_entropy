import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# Define a simple spectral model: linear combination of basis spectra
def build_basis(n_features=5, n_wavelengths=1000, seed=42):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 2500, n_wavelengths)  # nm
    bases = rng.normal(size=(n_features, n_wavelengths))
    return wavelengths, bases

# Generate synthetic spectra (true coefficients for each sample)
def generate_synthetic_spectra(bases, coeffs=None, noise_std=0.01, seed=123):
    rng = np.random.default_rng(seed)
    n_samples = coeffs.shape[0] if coeffs is not None else 10
    n_features, n_wavelengths = bases.shape
    if coeffs is None:
        coeffs = rng.normal(size=(n_samples, n_features))
    spectra = coeffs @ bases
    noise = rng.normal(scale=noise_std, size=spectra.shape)
    spectra += noise
    return spectra, coeffs

# create photometric filters: 
#   - 3 band (the 1st, 1response ?????????? 
def create_filters()..?