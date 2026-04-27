import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model (linear combination of basis spectra)
def build_basis(wavelengths, n_components=5):
    """Generate orthogonal basis spectra (e.g., Gaussian bumps)."""
    rng = np.random.default_rng(42)
    centers = rng.uniform(wavelengths[0], wavelengths[-1], n_components)
    widths = rng.uniform((wavelengths[-1]-wavelengths[0])/20,
                         (wavelengths[-1]-wavelengths[0])/10, n_components)
    amplitudes = rng.normal(size=n_components)
    basis = []
    for c, w, a in zip(centers, widths, amplitudes):
        spec = a * np.exp(-0.5 * ((wavelengths - c)/w)**2)
        basis.append(spec)
    return np.vstack(basis)  # shape (n_components, n_wavelengths)

def spectrum_from_coeffs(coeffs, basis):
    """Linear combination to produce a synthetic spectrum."""
    return coeffs @ basis  # shape (n_wavelengths,)

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_spectra, wavelengths, basis):
    """Create n_spectra random coefficients and corresponding spectra."""
    rng = np.random.default_rng(123)
    coeffs = rng.normal(size=(n_spectra, basis.shape[0]))
    spectra = coeffs @ basis.T  # shape (n_spectra, n_wavelengths)
    return coeffs, spectra

# 3. Generate photometric data from synthetic spectra
def photometric_response(wavelengths, filter_transmission):
    """Return a function that projects spectra onto filter response."""
    f = interp1d(wavelengths, filter_transmission, kind='linear',
                 bounds_error=False, fill_value=0.0)
    return lambda s: np.trapz(s * f(wavelengths), wavelengths)

def get_photometric_mags_spectrum(spectrum, filters):
    """Compute integrated fluxes for each filter band."""
    mags = []
    for filt in filters:
        mags.append(photometric_response(wavelengths, filt)(spectrum))
    return np.array(mags)

# 4. reconstruct a synthetic spectrum from photometric data
def reconstruct_spectrum_from_photometry(
        photometry, basis, wavelengths, filters, alpha=1.0):
    """
    Linearized reconstruction using ridge regression.
    -> Each filter flux is linear combination of basis integrals.
    We solve for coeffs that best match the photometric fluxes.
    1. Build matrix A where A[i,j] = integral of basis[j] * filt[i].
    A shape (n_filters, n_basis).
    1a.. If more filters than bases -> 2nd order. 
      <=>.. 
    ...
    """
continue the rest?? They want entire code. Let's finish.assistantfinal