import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# 1. Define a spectral model: linear combination of Gaussian basis functions
def gaussian_basis(wl, mu, sigma):
    """Return Gaussian basis function evaluated at wavelength array wl."""
    return np.exp(-(wl - mu)**2 / (2 * sigma**2))

def build_spectral_basis(wavelengths, mus, sigma):
    """Build design matrix for spectral basis given wavelengths and center points."""
    basis = np.array([gaussian_basis(wavelengths, mu, sigma) for mu in mus]).T
    return basis

def spectral_model(coeffs, basis):
    """Compute synthetic spectrum from basis coefficients."""
    return basis @ coeffs

# 2. Generate synthetic spectra
def generate_synthetic_spectra(num_spectra, wavelengths, mus, sigma):
    """Generate a set of synthetic spectra using random coefficients."""
    num_bases = len(mus)
    coeffs = np.random.randn(num_spectra, num_bases)
    basis = build_spectral_basis(wavelengths, mus, sigma)
    spectra = coeffs @ basis.T
    return spectra, coeffs, basis

# 3. **Mimic photometric data**
def photometric_filter(wl, center, width):
    # fuzzy
    n = 1..???
   ???.??..??????????



We need to produce final code with full implementation. Let's craft code:

- Define spectral model: Gaussian basis as above.

- Generate synthetic spectra.

- Generate photometric data: integrate spectra over filter transmission curves (boxcar or Gaussian). We'll implement filter transmission as Gaussian centered at bandpass center.

- Reconstruct spectrum from photometry: use regression from photometric fluxes to spectral coefficients using ridge regression, with basis projected onto filter responses.

Let's implement functions:

- build_filters(band_centers, band_widths, wavelengths): returns filter transmission arrays.

- compute_photometry(spectra, filters): return flux vector.

- reconstruct_spectrum_from_photometry(photometry, filters, basis): we can solve for coefficients by least squares: photometry = filters @ (basis @ coeffs) where filters shape (num_filters, num_wavelengths). So we want coeffs = argmin || phot - F B c||^2 + alpha ||c||^2. Solve with Ridge.

Implement:

- flatten: Use F B matrix = filters @ basis.

- Use Ridge(alpha=1e-3).fit(FB, photometry).coefficients.

- Then compute spectrum.

Let's implement a main routine generating data and performing reconstruction, comparing true vs reconstructed spectra.

Include minimal printing.

Let's produce code accordingly. Let's incorporate constraints: use numpy, scipy, scikit-learn.

Let's finalize.

Remember not to use any interactive reasoning or explanation.

Also no extra functions not used.

Return only code. Let's write final script.assistantfinal code