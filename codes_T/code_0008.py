import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
# 1. Define a simple spectral model: black‑body + Gaussian absorption line
# --------------------------------------------------------------------------- #

def black_body(wavelength, T):
    """Planck function in arbitrary units."""
    h = 6.62607015e-34  # Planck constant
    c = 2.99792458e8    # speed of light
    k = 1.380649e-23    # Boltzmann constant
    λ = wavelength * 1e-9
    B = (2 * h * c**2) / (λ**5 * (np.exp(h * c / (λ * k * T)) - 1))
    return B

def gaussian_absorption(wavelength, center, width, depth):
    """Gaussian absorption line."""
    return 1 - depth * np.exp(-((wavelength - center)**2)/(2 * width**2))

def spectral_model(wavelength, params):
    """
    Parameters:
        params[0] : Effective temperature (K)
        params[1] : Line center (nm)
        params[2] : Line width (nm)
        params[3] : Line depth (0-1)
    """
    T, center, width, depth = params
    flux = black_body(wavelength, T) * gaussian_absorption(
        wavelength, center, width, depth
    )
    return flux

# --------------------------------------------------------------------------- #
# 2. Generate synthetic spectra
# --------------------------------------------------------------------------- #

def generate_synthetic_spectra(num=50, seed=0):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 800, 400)  # nm
    spectra = []

    for _ in range(num):
        T = rng.uniform(5000, 6000)
        center = rng.uniform(550, 650)
        width = rng.uniform(2, 5)
        depth = rng.uniform(0.1, 0.4)
        spec = spectral_model(wavelengths, [T, center, width, depth])
        spectra.append(spec)

    return wavelengths, np.array(spectra)

# --------------------------------------------------------------------------- #
# 3. Generate photometric data from synthetic spectra
# --------------------------------------------------------------------------- #

def integrate_flux_in_band(wavelengths, flux, band_center, band_width):
    mask = (w_flux := (wavelengths >= band_center - band_width/2) &
            (wavelengths <= band_center + band_width/2))
    if np.any(mask):
        return np.trapz(flux[mask], wavelengths[mask]) / (band_width)
    else:
        return 0.0

def photometric_data_from_spectra(
    spectra, wavelengths, band_centers, band_width=20
):
    num_spectra = spectra.shape[0]
    photometry = np.zeros((num_spectrum, 3))
    for i, spec in enumerate(spectra):
        phot = [integrate_flux_in_band(
            wavelengths, spec, center, band_width) for center in band_centers]
        photometry[i] = phot
    return photometry

# --------------------------------------------------------------------------- #
# 1 & 2 : Define & create two synthetic spectral parameters
#   (for a small reference set and test data 
# ---------------------------------------------------------------------------  

t=



It seems the user wants a minimal implementation of core reconstruction framework. We need to produce runnable Python code that defines spectral model, generates synthetic spectra, generates photometric data, reconstruct a synthetic spectrum from photometric data. Use numpy, scipy, scikit-learn. Provide clear functions. No LOOCV etc.

We need to correct the incomplete script from assistant. Provide full script. Use a reference library of synthetic spectra and photometry to fit model parameters from photometry. Use regression to map photometry to spectrum. For simplicity, we can use linear regression (Ridge). We'll have training data: synthetic spectra and photometric values. Then reconstruct: given new photometric data, predict spectrum using regression. We need to produce synthetic example data within script.

Simpler approach: Use PCA or direct regression? We'll do: fit ridge regression to predict each flux point from photometric data. That is, model: flux vector (n_wavelength) predicted from photometry vector (n_band). Use Ridge regression on each wavelength independently. Then we can reconstruct new spectrum.

Implement functions: generate_synthetic_spectra(num), photometric_data_from_spectra(...). training: train models. reconstruction: reconstruct_spectrum(photometry, models). Provide example.

Let's write code.

Also fix earlier mistakes: variable names etc.

We'll produce a script with main block that demonstrates generation of training data, training, generating test photometry, reconstructing spectra.

Implementation details:

- wavelengths: array length M, e.g., 300 wavelengths between 400-800 nm.
- Band centers: maybe 3 bands: 450, 550, 650 nm, width 40 nm.
- Generate N training spectra (e.g., 100).
- For each training spectrum, compute photometric fluxes.
- Train Ridge regression models for each wavelength index: fit weights W_i such that flux_i = X * w_i + intercept.
- For reconstruction: given new photometry vector (3), compute predicted flux at each wavelength via dot product with trained weights + intercepts.

Return final script.

Let's craft the code accordingly.

Check errors: Use numpy functions correctly. Use scikit-learn's Ridge.

Edge cases: Use random seeds.

Now produce final answer with only code. Ensure no extraneous commentary.

Let's produce.assistantfinal code