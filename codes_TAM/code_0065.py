import numpy as np
from scipy import interpolate
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: linear combination of basis spectra
def generate_basis_spectra(n_basis, n_wavelengths, seed=0):
    rng = np.random.default_rng(seed)
    # random smooth spectra
    wavelengths = np.linspace(0, 1, n_wavelengths)
    basis = np.array([rng.normal(size=n_wavelengths) for _ in range(n_basis)])
    for i in range(n_basis):
        basis[i] = np.convolve(basis[i], np.ones(5)/5, mode='same')
    return wavelengths, basis

# 2. Generate synthetic spectra from random coefficients
def generate_synthetic_spectra(basis, n_samples, seed=1):
    rng = np.random.default_rng(seed)
    n_basis = basis.shape[0]
    coeffs = rng.uniform(-1, 1, size=(n_samples, n_basis))
    spectra = coeffs @ basis
    return coeffs, spectra

# 3. Simulate photometry: integrate spectra over filter curves
def simulate_photometry(spectra, wavelengths, filter_curves, noise_sigma=0.01):
    n_samples, n_wl = spectra.shape
    n_filters = filter_curvewidths.shape[0]
    fluxes = np.zeros((n_samples, n_filters))
    for f in range(n_filters):
        filt = filter_curves[f]
        fluxes[:, f] = spectra[:, :] @ filt
    return fluxes
def filter_curve(name, wavelengths):
    """ Simple Gaussian filter centered at a given wavelength fraction. """
    center = {'u':0.35,'g':0.45,'r':0.55,'i':None,'f':''.join???.. 
We need to produce code that meets requirements. Let's design:

- Define basis spectra: maybe 5 basis vectors over 200 wavelengths uniformly spaced.
- Synthetic spectra: random coefficients * basis + noise maybe.
- Photometry: define filter curves e.g., Gaussian filters; compute flux via dot product integral approx.

- Reconstruction: Use ridge regression to predict spectrum from photometry. Fit model on training data and then reconstruct.

We need synthetic data generation inside script: generate basis, synthetic spectra, photometry, then use reconstruction algorithm to recover spectrum.

Let's produce code that can run.

Implement functions:

- generate_basis_spectra(n_basis, n_wavelengths): returns wavelengths array and basis matrix shape (n_basis, n_wavelengths)

- generate_synthetic_spectra(basis, n_samples): returns coefficients matrix (n_samples, n_basis) and spectra (n_samples, n_wavelengths)

- build_filter_curves(wavelengths, n_filters): create list of filter response arrays (len(n_filters), len(wavelengths)), e.g., Gaussian centered at different fractions.

- simulate_photometry(spectra, filter_curves): compute fluxes by integrating spectra times filter curve; use trapezoidal integration.

- reconstruct_spectrum_from_photometry(photon_fluxes, filter_curves, wavelengths): use ridge regression: fit coefficients mapping from photometry to spectrum. We can construct matrix X (photon_fluxes) and Y (spectra). Fit ridge with alpha=1.0. Then predict spectra.

- main function to tie together: generate basis, synthetic spectra, filter curves, photometry, reconstruct, evaluate mean squared error.

Let's write code accordingly. Use numpy, scipy interpolate if needed but not required. Use scikit-learn's Ridge.

Return only code. Ensure no extra commentary. Let's produce final code.assistantfinal code