import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a spectral model: a Gaussian mixture representing a star's SED
def spectral_model(wavelengths, params):
    """
    wavelengths : array-like, shape (N,)
    params : list of tuples [(amp, cen, sigma), ...]
    returns : flux array (N,)
    """
    flux = np.zeros_like(wavelengths)
    for amp, cen, sigma in params:
        flux += amp * np.exp(-0.5 * ((wavelengths - cen) / sigma)**2)
    return flux

# 2. Generate synthetic spectra
def generate_synthetic_spectra(num_spectra, wavelength_grid):
    """
    num_spectra : int
    wavelength_grid : array-like, shape (N,)
    returns: array of shape (num_spectra, N)
    """
    spectra = []
    rng = np.random.default_rng()
    for _ in range(num_spectra):
        # Randomly choose 2 to 4 Gaussian components
        n_comp = rng.integers(2, 5)
        comps = []
        for _ in range(n_comp):
            amp = rng.uniform(0.5, 1.5)
            cen = rng.uniform(wavelength_grid.min(), wavelength_grid.max())
            sigma = rng.uniform(10, 50)
            comps.append((amp, cen, sigma))
        flux = spectral_model(wavelength_grid, comps)
        flux /= flux.max()  # normalize
        spectra.append(flux)
    return np.array(spectra)

# 3. Generate photometric data from synthetic spectra
def compute_photometry(spectrum, wave, filters):
    """
    spectrum : array-like (N,)
    wave : array (N,)
    filters : dict with keys 'name', 'transmission' (wave grid + trans)
    returns: dict of band fluxes
    """
    phot = {}
    for name, filt in filters.items():
        # integrate flux * filter trans over wavelength.
        trans = filt['trans']
        trans_interp = interp1d(wave, [0, 0] + trans[1:-1] + [0], kind='cubic')
?We need to finish implementation. Let's craft full code. Provide synthetic filters, etc. Let's build functions for interpolation. Also reconstruction: we have photometric measurements (band fluxes). We want to reconstruct spectrum using linear combination of basis spectra or via regression: ridge regression on basis set of synthetic spectra. Approach: create basis set of spectra (maybe principal components). Then solve for coefficients that match photometry. Use training set? But we can treat each spectrum as target and try to recover with ridge regression on photometric features to predict spectrum.

Simpler: Use training data: generate many spectra and their photometric colors, fit ridge regression to map photometric to spectrum values at each wavelength point separately. That is, for each wavelength pixel, train ridge regressor with features being photometric fluxes. Then for new photometric measurement, predict spectrum.

Let's implement: generate training set of synthetic spectra and photometry. Train ridge models per wavelength point. Reconstruct new synthetic spectrum.

We'll generate some random filters: say 5 broad-band filters with gaussian transmissions.

Ok write functions:

- generate_filters: returns dict of filter names -> {'wave': array, 'trans': array}

But filters used for computing photometry may need integration over same wavelength grid; we can use simple rectangular filters.

Simplify: Filters defined by central wavelength and width. Transmission is top-hat. We'll compute photometric flux via dot product of spectrum * transmission * delta_wavelength.

Implementation: compute_photometry(spectrum, wave, filters) returns dict.

Reconstruction: function reconstruct_spectrum_from_photometry(photometry, filters, wave, ridge_params=None)

Inside: training data: For each training sample, compute photometry and known spectrum. Build matrix X (samples x bands) and Y (samples x N_wave). Train ridge regressors for each wavelength. Predict Y_pred for new photometry.

Return predicted spectrum.

Also generate test synthetic spectrum, compute photometry, reconstruct.

Let's craft final code accordingly.

Need to ensure reproducibility.

Let's write code.

Use numpy arrays. For ridge, use alpha=1.0.

We'll define main script to demonstrate.

Ok. Let's code.assistantfinal code