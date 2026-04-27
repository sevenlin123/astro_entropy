import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a spectral model: simple linear combination of basis spectra
def generate_basis_spectra(wavelengths, n_components=5):
    """
    Generate synthetic basis spectra (e.g., Gaussian features).
    """
    np.random.seed(42)
    basis = []
    for _ in range(n_components):
        amp = np.random.uniform(0.5, 1.5)
        cen = np.random.uniform(wavelengths.min(), wavelengths.max())
        wid = np.random.uniform(10, 50)
        spec = amp * np.exp(-0.5 * ((wavelengths - cen) / wid) ** 2)
        basis.append(spec)
    return np.vstack(basis)

def build_model(wavelengths, n_components=5):
    """
    Build a linear model using the basis spectra.
    """
    basis = generate_basis_spectra(wavelengths, n_components)
    # Basis matrix shape: (n_components, n_wavelengths)
    return basis.T  # transpose to match (n_samples, n_features)

# 2. Generate synthetic spectra
def synthesize_spectrum(basis_matrix, coeffs=None):
    """
    Create a synthetic spectrum by linearly combining basis spectra.
    """
    if coeffs is None:
        coeffs = np.random.uniform(0.1, 1.0, size=basis_matrix.shape[1])
    spectrum = basis_matrix @ coeffs
    return spectrum, coeffs

# 3. Generate photometric data from synthetic spectra
def photometric_filter(wavelengths, filter_wave, width=30):
    """
    A simple top‑hopping filter centered at filter_wave.
    """
    filt = np.exp(-(w_flux->???)
    # ... (rest omitted because of truncated text)