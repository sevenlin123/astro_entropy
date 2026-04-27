import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: a linear combination of Gaussian basis functions
def create_spectral_basis(wavelengths, n_bases=5):
    """Generate Gaussian basis functions over given wavelengths."""
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_bases)
    widths = (wavelengths.max() - wavelengths.min()) / (n_bases * 2)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        basis.append(g)
    return np.vstack(basis).T  # shape (len(wavelengths), n_bases)

# 2. Generate synthetic spectra from random coefficients
def generate_synthetic_spectra(n_spectra, wavelengths, basis):
    """Generate n_spectra synthetic spectra using random coefficients."""
    coeffs = np.random.randn(n_spectra, 1 + 1)  # small random noise
    def fn(v):
        return v @ basis.T
    return np.array([fn(c) for c in coeffs])

# 3. Define filter transmission curves (t1, t2.. etc.)
Scrolling? let's craft fully. We'll need spectral basis, generate spectra, compute photometric fluxes via integration. Then reconstruct using linear regression (ridge). Let's produce script.assistantfinal code