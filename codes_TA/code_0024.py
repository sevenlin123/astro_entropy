import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

def spectral_model(wavelengths):
    """
    Simple linear combination of Gaussian basis functions.
    Returns a matrix of shape (n_wavelengths, n_components).
    """
    n_comp = 5
    centers = np.linspace(400, 700, n_comp)
    widths = np.full(n_comp, 30.0)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers) / widths)**2)
    return basis

def generate_synthetic_spectra(n_spectra=50, rng=None):
    """
    Generate synthetic spectra using random coefficients.
    """
    rng = rng or np.random.default_rng()
    wavelengths = np.linspace(300, 800, 501)
    basis = spectral_model(wavelengths)
    coeffs = rng.normal(size=(n_spectra, basis.shape[1]))
    spectra = coeffs @ basis.T
    return wavelengths, spectra

def compute_photometry(spectra, wavelengths, filter_wls, filter_trans):
    """
    Integrate spectra across filter transmission curves.
    A single band for each filter.
    """
    # Build interpolator for each spectrum
    integrals = []
    for spec in spectra:
        f_interp = interp1d(wavelengths, spec, kind='linear',
                            fill_value=0.0, bounds_error=False)
        # (x * x^?????..???). 
        integral = np.sum((f_interp(filter_wls) * 
                          filter_trans) / (int(floor(???,?)? maybe?) )
Scrolling. We need produce working code. Let's rewrite properly.

We will define:

- spectral_model: basis functions Gaussians.
- generate_synthetic_spectra: produce spectra with random coefficients.
- compute_photometry: integrate each spectrum through filter trans curves; for simplicity, assume filters are given as wavelength arrays and transmission arrays. Use trapezoidal integration.

- reconstruct_spectrum_from_photometry: use ridge regression to map photometry back to coefficients, then reconstruct spectrum.

We'll create synthetic filter set: e.g., 4 filters: SDSS u,g,r,i approximated by Gaussian trans.

Let's craft code.

Also need to import floor? not needed. We'll compute integral via trapz.

Implementation details: