import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def spectral_model(wavelengths, params):
    """
    Simple Gaussian spectral model.
    wavelengths: array of wavelengths
    params: [amplitude, center, width]
    Returns flux at each wavelength.
    """
    A, mu, sigma = params
    return A * np.exp(-(wavelengths - mu) ** 2 / (2 * sigma ** 2))

def generate_synthetic_spectra(n_spec, wavelengths):
    """Generate n synthetic spectra with random Gaussian parameters."""
    spectra = []
    rng = np.random.default_rng()
    for _ in range(n_spec):
        params = rng.uniform(low=[0.5, 4000., 200.], high=[1.5, 6000., 500.])
        spectral = spectral_model(wavelengths, params)
        spectra.append(spectral)
    return np.array(spectra)

def photometry_from_spectrum(spectrum, wavelengths, filters):
    """
    Convolve spectrally with a filter transmission curve.
    filter: 2D array (n_filters, n_wavelengths)
    returns fluxes in each band.
    """
    fluxes = []
    for filt in filters:
        conv = spectrum * filt
        flux = np.trapz(conv, wavelengths)
        # normalize by filter area
        norm = np.trapz(filt, wavelengths)
        fluxes.append(float(flux / norm))
    return np.array(fluxes)

def generate_filters(n_filters, wavelengths):
    """Create synthetic Gaussian filters centered between 4000-8000 Å."""
    rng = np.random.default_rng()
    filters = []
    for _ in *range(n_filters),:
        # ??? not good? let's just generate normal but
...