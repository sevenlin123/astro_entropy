import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

def spectral_model(wavelengths, params):
    """
    Simple spectral model: linear combination of basis functions.
    wavelengths: array of wavelengths
    params: array of coefficients for each basis function
    Returns: flux array
    """
    # Define basis: Gaussian peaks at 400, 500, 600 nm
    centers = np.array([400, 500, 600])
    widths = np.array([20, 20, 20])
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for coeff, center, width in zip(params, centers, widths):
        spectrum += coeff * np.exp(-((wavelengths - center) ** 2) / (2 * width**2))
    return spectrum

def generate_synthetic_spectra(num_spectra=10, wavemin=350, wavemax=750, nwav=400):
    """
    Generates synthetic spectra using random coefficients.
    1. Randomly generate coefficients for each spectrum.
    2. Compute fluxes using spectral_model.
    """
    wavelengths = np.linspace(wavelengths[0???]????? ???????) ????????? ????????...
"""
We need to write code accordingly. The previous snippet is broken; we need to craft full code. Let's create functions: generate_synthetic_spectra, generate_photometric_data, reconstruct_spectrum. We'll use a simple photometric filter set: e.g., U,B,V,R,I with central wavelengths and bandwidth. For each synthetic spectrum, integrate flux over filters to get photometric points. Then use ridge regression to fit coefficients given photometric measurements. Use training set of synthetic spectra to learn mapping from photometry to parameters, then reconstruct. Implementation steps:

- Create wavelength grid.
- Generate many synthetic spectra by random coefficients for basis functions.
- Compute photometric magnitudes: flux integrated over filter response. Use simple top-hat filters.

- Build matrix X_phot: shape (n_spectra, n_filters). Y_params: shape (n_spectra, n_basis).

- Train ridge regression: fit X_phot to Y_params.

- Reconstruct new synthetic spectrum: choose some params, compute true spectrum, compute its photometry, predict params via regression, reconstruct spectrum via spectral_model.

We need functions:

- make_filter_response(wavelengths, center, width): returns transmission.

- compute_photometry(spectrum, wavelengths, filters): integrate spectrum*filter / integral of filter.

We'll produce synthetic example: generate 100 spectra for training, 5 test spectra.

Now code. Ensure imports. Write functions.

Let's craft final code.assistantfinal code