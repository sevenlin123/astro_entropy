import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ----------------------------------------------------
# 1) Spectral model (a simple linear combination of basis spectra)
# ----------------------------------------------------
def spectral_model(x, coeffs, basis):
    """Compute a spectrum as a linear combination of basis spectra."""
    return coeffs @ basis(x)

# ----------------------------------------------------
# 2) Generate synthetic spectra
# ----------------------------------------------------
def generate_basis(n_basis, n_points, rng=None):
    """Generate random basis spectra."""
    rng = rng or np.random.default_rng()
    # Basis spectra are smooth random functions over wavelength range [400, 800] nm
    wavelengths = np.linspace(400, 800, n_points)
    basis = np.zeros((n_basis, n_points))
    for i in range(n_basis):
        # Random polynomial coefficients for smoothness
        coeffs = rng.normal(size=4)  # cubic polynomial
        poly = np.polyval(coeffs, (wavelengths - 600) / 200)  # shift/scale
        basis[i] = np.exp(-0.5 * ((poly - 0.5) ** 2)) + rng.uniform(0, 0.1, n_points)
    return wavelengths, basis

def create_synthetic_spectrum(coeffs, wavelengths, basis):
    """Build a synthetic spectrum by combining basis spectra."""
    return coeffs @ basis

# ----------------------------------------------------
# 3) Photometric data from synthetic spectra
# ----------------------------------------------------
def bandpass_filter(wl, flux, central, width):
    # Gaussian bandpass centered at 'central'
    filt = np.exp(-(wl - central)**2 / (2 * width**2))
    return np.trapz(flux * filt, wl) / np.trapz(filt, wl)

def generate_photometry(spectrum, wl, band_centers, band_widths):
    """Compute synthetic photometric points (AB magnitude).
    1: flux density per unit wavelength.
    """
    phot = []
    for c, w in zip(band_centessc, band_widths):
        flux_avg = bandpass_filter(wl, spectrum, c, w)
        phot.append(flux_avg)
    return np.array(phot)

def calc_ab_mag(flux, lam=700????????????????????



We have to produce runnable Python code implementing core reconstruction framework. The previous snippet incomplete.

We must output final code only, no explanation. Let's design entire script.

We need:

- Define spectral model: linear combination of basis spectra.

- Generate synthetic spectra.

- Generate photometric data from synthetic spectra.

- Reconstruct a synthetic spectrum from photometric data.

Use numpy, scipy, scikit-learn.

Implementation details:

- Basis spectra: array shape (n_basis, n_wavelengths). We'll create with random smooth functions, maybe via Gaussian random fields.

- Synthetic spectrum: choose random coefficients vector length n_basis.

- Photometric data: compute bandpasses: integrate flux over filter transmission.

- Reconstruction: Use ridge regression to solve for coefficients given photometric measurements and basis integrated over filters.

Simplify: We can precompute basis filter integrals: For each band, compute integral of each basis spectrum through that band. Then we have linear relation photometric vector = A * coeffs where A is matrix (n_bands x n_basis). Solve for coeffs via ridge or linear least squares. Then reconstruct spectrum.

Let's craft code.

Functions:

- generate_wavelengths(n_points): returns array.

- generate_basis(n_basis, wavelengths): returns basis matrix.

- synth_spectrum(coeffs, basis): dot product.

- bandpasses: list of tuples (center, width). Create integration functions.

- bandpass_integral(basis, wavelengths, center, width): integrate each basis spectrum weighted by filter transmission. Returns array of length n_basis.

- photometry_from_spectrum(spectrum, wavelengths, bands): compute flux for each band.

- reconstruct_coeffs(photometry, band_integrals): solve linear system via Ridge regression or np.linalg.lstsq.

- reconstruct_spectrum(coeffs, basis): produce flux.

Test:

- set random seed.

- n_basis = 5, n_points = 1000, n_bands = 4.

- Generate basis.

- Generate random coefficients.

- Create synthetic spectrum.

- Compute photometry.

- Build band_integrals matrix (n_bands x n_basis).

- Reconstruct coefficients.

- Compare original vs reconstructed.

Print mean squared error.

Ok.

Edge cases: For integration use trapz.

Define Gaussian filter.

Will also import necessary modules.

Return code.

Let's write full code accordingly.

Remember to only output code, no explanations. Also earlier they had some syntax errors; we will correct.

Ok produce final code.assistantfinal code