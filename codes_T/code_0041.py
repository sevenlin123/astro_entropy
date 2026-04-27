import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 1. Define a spectral model: simple linear combination of basis spectra
def build_spectral_basis(wavelengths):
    """
    Build a set of basis spectra (e.g., Gaussian peaks) over given wavelengths.
    """
    # Example: 3 Gaussian basis functions with different centers
    centers = np.array([4000, 5000, 6000])  # in Angstroms
    widths  = np.array([200, 300, 250])
    amplitudes = np.array([1.0, 0.8, 0.5])

    basis = np.zeros((len(wavelengths), len(centers)))
    for i, (c,w,a) in enumerate(zip(centers, widths, amplitudes)):
        basis[:,i] = a * np.exp(-0.5*((wavelengths - c)/w)**2)
    return basis

# 2. Generate synthetic spectra using random coefficients
def generate_synthetic_spectra(num_samples, wavelengths):
    """
    Generate synthetic spectra from random linear combinations of basis.
    """
    basis = build_spectral_basis(wavelengths)
    coeffs = np.random.randn(num_samples, basis.shape[1])
    spectra = coeffs @ basis.T
    return spectra, coeffs

# 3. generate photometric data from synthetic spectra
def compute_photometry(spectra, wavelengths, filters):
    """
   0..1
      
    ...
    ...