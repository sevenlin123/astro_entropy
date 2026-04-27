import numpy as np
from scipy.special import legendre
from sklearn.linear_model import Ridge

# ----- 1. Define a spectral model ------------------------------------------
def spectral_model(coeffs, wavelengths):
    """
    Evaluate a synthetic spectrum as a linear combination of Legendre polynomials.
    Parameters
    ----------
    coeffs : array_like, shape (Nbasis,)
        Coefficients of the Legendre basis.
    wavelengths : array_like, shape (M,)
        Wavelength points where the spectrum is evaluated.
    Returns
    -------
    flux : ndarray, shape (M,)
        The model flux at each wavelength.
    """
    # Normalise wavelength to [-1, 1] for Legendre polynomials
    wl_norm = (wavelengths - wavelengths.mean()) / (0.5 * wavelengths.max() + 0.5 * wavelengths.min())
    flux = np.zeros_like(wl_norm)
    for i in range(len(coeffs)):
        flux += coeffs[i] * legendre(i)(wl_norm)
    return flux

# ----- 2. Generate synthetic spectra ----------------------------------------
def generate_synthetic_spectra(num_spectra, num_pixels, coeff_range=(0.1, 1.0), seed=42):
    rng = np.random.default_rng(seed)
    spectra = []
    coeffs_list = []
    for _ in range(2 * num_spectra):
        coeffs = rng.uniform(coeff_range[0], coeff_range[2] if len(coeff_range) > 1 else coeff_range[1], size=num_pixels)
        # (i) ... (ii)