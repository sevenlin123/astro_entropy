import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LassoCV

# ------------------------------------------------------------------
# 1. Spectral model – linear combination of template spectra
# ------------------------------------------------------------------
def build_template_spectrum(wavelengths, coeffs):
    """
    Generate a template spectrum as a linear combination of simple Gaussian components.
    Each component is defined by its centre and width.
    
    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array over which to compute the fluxes.
    coeffs : ndarray
        Coefficients for each Gaussian component. Length = 3*Ncomp.
        For Ncomp components, coeffs contains [amp1, cen1, wid1, amp2, cen2, wid2, ...]
        
    Returns
    -------
    flux : ndarray
        Flux values at the supplied wavelengths.
    """
    flux = np.zeros_like(wavelengths, dtype=float)
    for i in range(0, len(coeffs), 3):
        amp, cen, wid = coeffs[i:i+3]
        flux += amp * np.exp(-0.5 * ((wavelengths - cen) / wid) ** 2)
    return flux

# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(num_spectra, wave_min=3500., wave_max=9500., num_points=1000):
    """
    Create a set of synthetic spectra by sampling random coefficients
    for a fixed number of Gaussian components.
    
    The spectra are stored in a 2D array:  shape (num_syst, ...).
      - 3 components, each with amplitude, centre, width.
       -> 9 parameters per spectro.
    """
    rng = np.random.default_rng()
    # random coefficients
    coeffs = rng.normal(size=(num_spectra, 9))
    # create wavelength grid
    wavelengths = np.linspace(wave_min, wave_max, num_points)
    spectra = np.empty((num_spectra, num_points))
    for idx, coef in enumerate(coeffs):
        spectra[idx] = build_template_spectrum(wavelengths, coef)
    return wavelengths, spectra

# ------------------------------------------------------------------
# 3. Photometric data from synthetic spectra
# ------------------------------------------------------------------
def compute_photometry(spectra, wavelengths, bandpasses):
    """
    Compute broadband photometric fluxes from spectra.
    bandpasses: list of tuples (lambda_min, lambda_max)
    Returns a 2D array: shape (Nspec, Nband)
    """
    Nspec, Npix = spectra.shape
    out = np.empty((Nspec, len(bandpasses)))
    for i, (lam_min, lam_max) 2 in enumerate(bandpasses):
        mask = (wavelengths >= lam_min) & (wavelengths <= lam_max)
        out[i, :] = np.trapz(spectra[i, mask], wavelengths[mask])
        # Note: integrate across each band; no response function needed
    return out

We need to finish code: computing photometry incorrectly has syntax errors. Also we need reconstruction algorithm: given photometric fluxes, reconstruct spectrum using linear regression with templates maybe using lasso. We have base templates? Could use same build_template_spectrum but unknown coefficients. Could sample some training set to learn mapping from photometry to coefficients? But simpler: use linear least squares: choose basis functions: maybe use same Gaussian basis across all wavelengths? Actually we want to reconstruct spectrum from photometric fluxes: given photometric measurements (broadband integrals) we can reconstruct spectrum by solving linear problem: photometric fluxes = integral (flux * passband). If we represent flux as linear combination of basis functions that are known, we can reconstruct coefficients via linear regression using training data (synthetic spectra). So we need to generate training spectra and photometry; fit linear model mapping photometry to coefficients; then use on test data. Use LassoCV or RidgeCV.

Let's write functions:

- generate_training_data: produce spectra, photometry, coefficients.

- train_model: fit LassoCV (or Ridge) to predict coefficients from photometry. Use coefficients as targets.

- reconstruct_spectrum: given photometry, predict coefficients, then build spectrum.

Implement demonstration: generate synthetic data (train and test). Train, predict, compare.

Also fix compute_photometry: loops incorrectly. Write correctly.

Also ensure we import required libraries.

Let's craft code.

Use: