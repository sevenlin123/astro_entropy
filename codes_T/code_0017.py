import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1) Define a simple spectral model
def spectral_model(wavelengths, coeffs):
    """
    Linear combination of basis spectra.
    wavelengths: 1D array of wavelength values
    coeffs: array of coefficients for basis spectra
    Returns flux array corresponding to wavelengths.
    """
    # Example basis: Gaussian peaks at 400, 500, 600 nm with widths 20 nm
    gauss = lambda mu, sigma: np.exp(-0.5 * ((wavelengths - mu)/sigma)**2)
    base1 = gauss(400, 20)
    base2 = gauss(500, 20)
    base3 = gauss(600, 20)
    return coeffs[0]*base1 + coeffs[1]*base2 + coeffs[2]*base3

# 2) Generate synthetic spectra
def generate_synthetic_spectra(n_samples, wavelengths):
    """
    Generate n_samples synthetic spectra using random coefficients.
    Returns flux matrix (n_samples, len(wavelengths))
    """
    rng = np.random.default_rng()
    coeffs = rng.normal(size=(n_samples, 3))
    spectra = np.array([spectral_model(wavelengths, c) for c in coeffs])
    return coeffs, spectra

# 3) photometric data from synthetic spectra
def photometric_from_spectrum(spectra, filter_wavelengths, filter_trans):
    """
    Compute photometric measurements by integrating the spectrum over filter transmission.
    - spectra: (n_samples, n_wave) array
    - filter_wavelengths: 1D array of wavelengths defining filter response
    - filter_trans: 1D array of transmission values (same shape as filter_wavelengths)
    Returns photometry (n_samples, 1) array
    """
    # Interpolate each spectrum onto the filter grid
    phot = []
    for spec in spectra:
        f_interp = interp1d(wavelengths, spec, kind='linear', bounds_error=False, fill_value=0.0)
        interp_vals = f_interp(filter_wavelengths)
        integral = np.trapz(interp_vals * filter_trans, filter_wavelengths)
        phot.append(integral)
    return np.array(phot).reshape(-1,1)

# 4) Reconstruct a synthetic spectrum from photometric data
def reconstruct_from_photometry(photometry, wavelengths, filter_wavelengths, filter_trans):
    """
    Use ridge regression to fit a linear combination of basis spectra to match photometry.
    """
    # Build design matrix X (n_samples, n_basis)
    n_samples = photometry.shape[0]
    n_basis = 3
    X = np.zeros((n_samples, n_basis))
    for i in range(n_basis):
        # Basis function on filter grid
        basis = np.exp(-0.5*((filter_wavelengths - [400, 500, 600][i])/20)**f? ?????????? 
        ...