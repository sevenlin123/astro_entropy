import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV

def spectral_model(wavelengths, coeffs):
    """Linear combination of Gaussian basis functions."""
    centers = np.linspace(4000, 8000, len(coeffs))
    widths = np.full_like(centers, 200.0)
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for a, c, w in zip(coeffs, centers, widths):
        spectrum += a * np.exp(-0.5 * ((wavelengths - c) / w) ** 2)
    return spectrum

def generate_synthetic_spectra(n_samples, wavelengths):
    rng = np.random.default_rng()
    coeffs = rng.standard_normal(size=(n_samples, len(wavelengths)))
    spectra = np.array([spectral_model(wavelengths, c) for c in coeffs])
    return spectra, coeffs

def photometry_from_spectrum(spectra, filters, wavelengths):
    fluxes = []
    for filt in filters:
        # filter transmission curves (filt has shape [len(wavelengths)])
        interp_filt = interp1d(wavelengths, filt, bounds_error=False, fill_value=0.0)
        for spec in spectra:
            flux = np.trapz(spec * interp_filt(wavelengths), wavelengths)
            fluxes.append((spec, flux))
    return flux * np.array([flux[1] for flux in fluxes])

def create_filters(num_filters, wavelengths):
    rng = npwide
??