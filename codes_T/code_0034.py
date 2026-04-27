import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------------------------
# 1) Define a spectral model
# ------------------------------------
def spectral_model(wave_grid, params):
    """
    Simple model: combination of two Gaussian emission lines
    and a linear continuum.
    
    Parameters
    ----------
    wave_grid : array-like
        Wavelength grid (Angstrom).
    params : dict
        Dictionary with keys:
          'ampl1', 'center1', 'width1',
          'ampl2', 'center2', 'width2',
          'slope',  'intercept'
    Returns
    -------
    spec : ndarray
        Spectrum fluxes on the supplied wavelength grid.
    """
    g1 = params['ampl1'] * np.exp(
        -0.5 * ((wave_grid - params['center1']) / params['width1'])**2
    )
    g2 = params['ampl2'] * np.exp(
        -0.8 * ((wave_grid - params['center2']) / 
                (params['width2']))**2
    )
    cont = params['slope'] * wave_grid + 
            (dict=???)   # <--- This line does not create
The assistant truncated due to error. Need to produce full code correctly. Let's design: we need a spectral model function that produces synthetic spectra based on parameters. Then generate synthetic spectra for multiple samples. Then generate photometric data by integrating over filter transmission curves. Then reconstruct synthetic spectrum from photometry using linear regression (Ridge). Use numpy, scipy, sklearn.

We must produce complete runnable script.

Let's design steps:

- Define wavelengths grid: e.g., 4000-8000 Angstrom, step 1 A (4000 points).
- Define filter set: e.g., UBVRI filters: approximate throughput as Gaussian centered at certain wavelength and width. Create filter transmission arrays.

- For each sample (say 100), generate random parameters within ranges, compute spectrum.

- Compute synthetic photometry: integrate spectrum * filter / integral(filter). produce fluxes in each band.

- Reconstruction: Use training set (all except target? But we can use all to fit linear model mapping from photometric vector to spectrum). Actually we want to reconstruct spectrum from photometry; we can train a regression model that maps photometric values to full spectrum (vector). So features X: shape (n_samples, n_filters). Targets Y: shape (n_samples, n_wavelength). Use Ridge regression with alpha.

- After training, test on new synthetic sample: compute its photometry, predict spectrum, compare.

Implement functions:

- get_synthetic_params(n): return array of parameter dictionaries.

- generate_spectrum(params): returns flux array.

- compute_photometry(spectra, filters): returns photometric flux per filter.

- reconstruct_spectrum(X, Y): train ridge, return model.

- predict_spectrum(model, photometry): predict.

- main: generate synthetic dataset, train, test.

Also need filter definitions: create dictionary of filter names mapping to transmission arrays.

Ok produce code.

Edge: ensure reproducibility: set seed.

Let's write code.assistantfinal code