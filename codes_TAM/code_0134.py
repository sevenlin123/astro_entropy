import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 1. Define a simple spectral model
def spectral_model(wavelengths, coeffs):
    """
    A linear combination of Gaussian basis functions.
    wavelengths: array-like, shape (n_points,)
    coeffs: array-like, shape (n_basis,)
    """
    n_basis = len(coeffs)
    base = []
    centers = np.linspace(4000, 7000, n_basis)  # nm
    widths = np.full(n_basis, 200.0)          # nm
    for c, w in zip(centers, widths):
        base.append(np.exp(-(wavelengths - c)**2 / (2 * w**2)))
    base = np.vstack(base).T  # shape (n_points, n_basis)
    return base @ coeffs

# 2. generate synthetic spectra
def generate_synthetic_spectra(n_samples=5, n_points=200, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    wavelengths = np.linspace(4000, 7000, n_points)  # nm
    true_coeffs = rng.normal(scale=1.0, size=(n_samples, 10))
    spectra = np.array([spectral_model(wavelengths, c) for c in true_coeffs])
    # normalize by total flux to have comparable intensities
    spectra /= spectra.sum(axis=1, keepdims=True)
    return wavelengths, spectra, true_coeffs

# 3. generate photometric data from synthetic spectra
def compute_photometry(spectra, wavelengths, filter_wavelengths, filter_transmissions):
    """
    Convolve spectra with filter transmissions.
    spectra: array-like (n_samples, n_points)
    filter_wavelengths: list of arrays
    filter_transmission: list of arrays
    returns: photometric fluxes (n_samples, n_filters)
    """
    photometry = []
    for filt_w, filt_t in zip(filter_filter_wavelengths, filter_filter_transmissions):
        f = np.trapz(spectra[:, None] * filt_t[None, :], x=wavelengths, axis=1)
        photometry.append(f)
    [???]