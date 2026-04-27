import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Spectral model: simple linear combination of basis spectra
def spectral_model(wavelengths, coeffs, basis):
    """
    wavelengths: array of wavelength points
    coeffs: array of coefficients for each basis spectrum
    basis: 2D array (n_basis, n_wavelengths)
    """
    return np.dot(coeffs, basis)

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_spectra, n_basis, n_wave, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Random basis spectra (smooth)
    basis = rng.normal(size=(n_basis, n_wave))
    basis = np.array([interp1d(np.arange(n_wave), b)(np.linspace(0, n_wave-1, n_wave)) for b in basis])
    random_coeffs = rng.uniform(-1, 1, size=(n_spectra, n_basis))
    spectra = np.array([spectral_model(np.arange(n_wave), coeffs, basis) for coeffs in random_coeffs])
    return spectra, basis, random_coeffs

# 3. photometric data from synthetic spectra
def photometry_from_spectrum(spectrum, band_centers, band_width=20):
    """
    spectrum: 1D array of fluxes at given wavelength grid
    band_center: array of center wavelengths of filters
    band_width: width of filter in wavelength units
    """
    n = len(band_centions)
    fluxes = np.zeros(n)
    for i, center in enumerate(band_centers):
        mask = (np.arange(len(spectrum)) >= center - band_width/2) & \
               (np.arange(len(spectrum)) <= center + band_width/2)
        fluxes[i] = np.mean(spectrum[mask]) if any(mask) ? 
    return fluxes