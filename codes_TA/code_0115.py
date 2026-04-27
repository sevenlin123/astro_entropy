import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge


def spectral_model(wavelengths, params):
    """
    Simple Gaussian spectral model with continuum.
    wavelengths: array of wavelength values
    params: dict with keys 'amplitude', 'center', 'width', 'continuum'
    """
    amp = params['amplitude']
    cen = params['center']
    wid = params['width']
    cont = params['continuum']
    return cont + amp * np.exp(-0.5 * ((wavelengths - cen) / wid) ** 2)


def generate_synthetic_spectra(n_spectra, wavelengths):
    """Generate a set of synthetic spectra using random parameters."""
    spectra = []
    for _ in range(n_spectra):
        params = {
            'amplitude': np.random.uniform(0.5, 1.5),
            'center': np.random.uniform(5000, 6000),
            'width': np.random.uniform(50, 150),
            'continuum': np.random.uniform(0.8, 1.2)
        }
        spectra.append(spectral_model(wavelengths, params))
    return np.array(spectra)


def photometry_from_spectrum(spectrum, wavelengths, filter_wavelengths, filter_response):
    """
    Compute synthetic photometry by integrating the product of the spectrum
    and filter response over wavelength.
    """
    interp_spec = interp1d(wavelengths, spectrum, kind='linear', fill_value='extrapolate')
    flux = np.trapz(
        interp_spec(filter_wavelengths) * filter_response,
        filter_wavelengths
    )
    return flux


def create_filters(???):