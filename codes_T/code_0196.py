import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Define a simple linear spectral model
def spectral_model(wavelengths, coeffs):
    """Generate a spectrum given wavelengths and linear coefficients."""
    return np.polyval(coeffs[::-1], wavelengths)  # coefficients in ascending order

# 2. Generate synthetic spectra
def generate_synthetic_spectra(num_spectra, wavelengths):
    """Create synthetic spectra with random linear coefficients."""
    coeffs = np.random.randn(num_spectra, len(wavelengths))
    spectra = np.array([spectral_model(wavelengths, c) for c in coeffs])
    return spectra, coeffs

# 3. Generate photometric data from synthetic spectra
def photometric_data(spectra, wavelengths, filters):
    """
    Simulate photometric measurements by integrating spectra over filter passbands.
    - spectra: array of shape (n_samples, n_wavelengths)
    - wavelengths: array of wavelength points
    - filters: list of filter transmission arrays, same shape as spectra
    """
    photometry = []
    for filt in filters:
        integrand = spectra * filt[None, :]  # broadcast
        band_flux = np.trapz(integral=integrand, x=wavelengths, axis=1)
        band_flux *= 0.1  # scaling to mimic non-dim. values
    return np.column_stack(photometry)

def add_noise(spectrum, noise_level=0.01):
    return spectrum + np.random.normal(scale=noise_to_i???

We have partial code. Need to finish implementation: generate filters, etc., reconstruct synthetic spectrum from photometry using linear regression with basis functions maybe. Provide functions.

Let's craft full code: define wavelengths array, generate synthetic spectra, generate filter responses maybe random Gaussian filters.

Photometric integration: use trapezoidal integration over wavelength.

Reconstruction: we can fit linear regression: we have known filter responses; we want to reconstruct spectrum as linear combination of basis functions? Perhaps simplest: use linear regression where inputs are photometric fluxes, outputs are spectra (wavelength grid). So train regression on training set (synthetic). Then predict for new photometry.

Implement functions:

- generate_filters(n_filters, wavelengths): random Gaussian passes.

- compute_photometry(spectra,wavelengths,filters): integrate.

- reconstruct_spectrum(photometry, filters, wavelengths, n_components=...): Fit regression on training data.

Use train-test split for demonstration.

Provide main block generating data, training, predicting, compare.

Let's write. Note earlier incomplete add_noise function: we ignore maybe.

Also need to compute band_flux per filter properly.

Let's write final code.assistantfinal code