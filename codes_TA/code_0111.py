import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# 1. Define a spectral model: a simple Gaussian mixture
def spectral_model(wavelengths, params):
    """
    wavelengths : array-like, shape (n_samples,)
    params : array-like, shape (n_components, 3)
             Each component is [amplitude, center, sigma]
    Returns: flux array of same shape as wavelengths
    """
    flux = np.zeros_like(wavelengths, dtype=float)
    for amp, cen, sig in params:
        flux += amp * np.exp(-0.5 * ((wavelengths - cen) / sig)**2)
    return flux

# 2. generate synthetic spectra
def generate_synthetic_spectra(n_spectra, wavelengths, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    spectra = []
    for _ in range(n_spectra):
        # Randomly choose number of components between 1 and 3
        n_comp = rng.integers(1, 4)
        # Random amplitudes (0.5 to 2.0), centers (5000-6000 A), sigma (50-200 A)
        amplitudes = rng.uniform(0.5, 2.0, size=n_comp)
        centers = rng.uniform(5000, 6000, size=n_comp)
        sigma = rng.uniform(50, 200, size=n_comp)
        params = np.column_stack((amplitudes, centers, sigma))
        flux = spectral_model(wavelengths, params)
        spectra.append((params, flux))
    return spectra

# 3. generate photometric data from synthetic spectra
def generate_photometry(spectra, filters):
    """
    spectra: list of (params, flux) tuples
    filters: list of filter transmission curves
    Returns: photometric fluxes array, shape (n_spectra, n_filters)
    """
    n_filters = len(filters)
    n_spectra = len(spectra)
    phot_flux = np.empty((n_spectrum, n_filters))
    for i, (_, flux) in enumerate(spectra):
        for j, filt in enumerate(filters):
            # Weighted average over wavelength with filter transmission
            filt_interp = interp1d(*filt, kind='linear', bounds_error=False, fill_value=0.0)
            trans = filt_interp(wavelengths)
            phot_flux[i, j] = np.trapz(flux * trans, wavelengths) / (np.tran...