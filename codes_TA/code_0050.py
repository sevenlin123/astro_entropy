import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, params):
    """
    Simple linear combination of Gaussian basis functions.
    Parameters: array-like of shape (n_basis,)
    """
    n_basis = len(params)
    spectrum = np.zeros_like(wavelengths, dtype=float)
    centers = np.linspace(4000, 8000, n_basis)  # Å
    widths = np.full(n_basis, 200.0)           # Å
    for i, p in enumerate(params):
        spectrum += p * np.exp(-0.5 * ((wavelengths - centers[i]) / widths[i]) ** 2)
    return spectrum

def generate_synthetic_spectra(num_spectra, num_pixels, rng=None):
    """
    Generate synthetic spectra by sampling random coefficients.
    """
    if rng is None:
        rng = np.random.default_rng()
    wavelengths = np.linspace(4000, 8000, num_pixels)
    coeffs = rng.normal(size=(num_spectra, 1))  # choose one parameter
    spectra = np.array([spectral_model(wavelengths, c) for c in coeffs])
    return wavelengths, spectra, coeffs

def gaussian_filter_band(wavelengths, center, width):
    # create a Gaussian filter transmission curve
    return np.exp(-(wavelengths * 0x01a + 
                   *???)"""

We need to finish the code; earlier incomplete. Let's design simple system: Use known filters: e.g., SDSS u,g,r,i,z with central wavelengths. We'll compute photometric flux by integrating spectrum * filter response. Then reconstruct using ridge regression on basis functions. Provide functions: generate_filters, compute_photometry, reconstruct_spectrum. Provide synthetic example and print error.

Let's craft.

Need to use numpy, scipy, sklearn. We'll create filter transmission curves as Gaussian.

Implement functions:

- filter_response(wavelengths, center, sigma)
- compute_photometry(spectra, wavelengths, filters)
- reconstruct_spectrum(photometry, wavelengths, filters, n_basis=10)

Use ridge regression to fit coefficients.

Let's write code accordingly. Return final runnable script. Ensure no incomplete code.

Let's produce final code.assistantfinal code