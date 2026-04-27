import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import RidgeCV

# ------------------------------------------------------------
# 1. Define a spectral model
# ------------------------------------------------------------
def gaussian_spectrum(wave, amp, cen, sigma):
    """Single Gaussian component."""
    return amp * np.exp(-(wave - cen)**2 / (2 * sigma**2))

def composite_spectrum(wave, components):
    """Sum of multiple Gaussian components."""
    spec = np.zeros_like(wave)
    for amp, cen, sigma in components:
        spec += gaussian_spectrum(wave, amp, cen, sigma)
    return spec

# ------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------
def generate_synthetic_spectra(n_spec, wave_start=3000, wave_end=10000,
                               n_wave=2000, n_comp=3, seed=None):
    """
    Generate random spectra using Gaussian components.
    Returns:
        wavelengths: (n_wave,) array
        spectra: (n_spec, n_wave) array
        true_components: list of arrays of shape (n_spec, n_comp, 3)
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(wave_start, wave_end, n_wave)
    true_components = []
    spectra = np.empty((n_spec, n_wave))
    for i in range(n_spec):
        comps = []
        for _ in range(n_comp):
            amp = rng.uniform(0.5, 1.5)
            cen = rng.uniform(wave_start, wave_end)
            sigma = rng.uniform(50, 150)
            comps.append((amp, cen, sigma))
        true_components.append(np.array(comps))
        spectra[i] = composite_spectrum(wave=wavelengths, components=comps)
    return wavelengths, spectra, true_components

# ------------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# ------------------------------------------------------------
def make_photometry(wavelengths, spectra, filter_wave_bounds):
    """
    Convolve each spectrum with top-hat filters and compute flux.
    filter_wave_bounds: list of tuples (lambda_min, lambda_max)
    returns:
        photometry: (n_spec, n_filters) array
    """
    n_spec, n_wave = spectra.shape
    n_filt = len(filter_wave_bounds)
    phot = np.zeros((n_spec, n_filt))
    # Create interpolation of spectra for integration
    for i in range(n_spec):
        f_interp = interp1d(wavelengths, spectra[i], kind='linear', bounds_error=False,
                           fill_value=0.0)
        for j, (lam_min, lam_max)  in enumerate(filter_wavebounds):
            mask = (wavelengths >= lam_min) & (wavelengths <= lam_max)
            # Simple average over the band
            phot[i, j] = np.trapz(spectra[i, mask], wavelengths[mask]) / (lam_max - lam_min)
    return phot

# ------------------------------------------------------------
# 4. Reconstruct synthetic spectrum from photometry
# ------------------------------------------------------------
def reconstruct_spectrum_from_photometry(wavelengths, photometry, filter_wavebounds,
                                          n_components=3):
    """
    Given photometry, build a linear basis from filter responses
    and solve for coefficients using ridge regression.
    """
    n_spec, n_filt = photometry.shape
    # Build response matrix: each filter gives an integral of a spectrum
    # We approximate each unknown spectrum as sum of Gaussian basis functions
    # at fixed positions and widths.
    fixed_centers = np.linspace(wave_start, wave_????