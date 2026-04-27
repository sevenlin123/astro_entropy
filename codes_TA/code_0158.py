#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.

The script defines a simple spectral model (linear combination of Gaussian
basis functions), generates synthetic spectra, creates photometric data from
these spectra using a few broad-band filters, and finally reconstructs a
spectrum from its photometric measurements via least‑squares inversion.
"""

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Spectral model
# --------------------------------------------------------------------------- #

WAVELENGTHS = np.linspace(400, 800, 1000)  # nm

def gaussian_basis():
    """Return a list of Gaussian basis functions."""
    centers = np.array([450, 520, 600, 680])   # nm
    widths = np.array([20, 30, 25, 15])        # nm
    basis = []
    for c, w in zip(centers, widths):
        basis.append(norm.pdf(WAVELENGTHS, loc=c, scale=w))
    return np.vstack(basis)  # shape (n_basis, n_wavelength)

BASIS = gaussian_basis()          # shape (4, 1000)
N_BASIS = BASIS.shape[0]

def generate_synthetic_spectrum(coeffs):
    """Linear combination of basis functions."""
    return coeffs @ BASIS      # shape (n_wavelength,)

# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

def gaussian_filter(center, width):
    """Return a Gaussian filter transmission curve."""
    return norm.pdf(WAVELENGTHS, loc=center, scale=width)

FILTERS = [
    gaussian_filter(500, 40),
    gaussian_filter(600, 50),
    gaussian_filter(700, 60),
]

N_FILTERS = len(FILTERS)

def compute_photometry(spectrum, filters=FILTERS):
    """Integrate spectrum over each filter."""
    phots = []
    for filt in filters:
        flux = np.trapz(spectrum * filt, WAVELENGTHS)
        phots.append(flux)
    return np.array(phots)   # shape (n_filters,)

# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #

np.random.seed(42)
N_SAMPLES = 10
coeff_matrix = np.random.uniform(low=0.5, high=1.5, size=(N_SAMPLES, N_BASIS))

spectra = np.array([generate_synthetic_spectrum(c) for c in coeff_matrix])
photometries = np.array([compute_photometry(s) for s in spectra])

# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

def build_design_matrix(filters=FILTERS, basis=BASIS):
    """
    For each filter, compute the response of every basis function.
    Returns a matrix A of shape (n_filters, n_basis) such that
        y = A @ coeffs
    """
    A = np.zeros((len(filters), basis.shape[0]))
    for i, filt in enumerate(filters):
        for j, b in enumerate(basis):
            A[i, j] = np.trapz(b * filt, WAVELENGTHS)
    return A

DESIGN = build_design_matrix()

def reconstruct_spectrum(photometry, design=DESIGN, basis=BASIS):
    """Recover coefficients via least squares and rebuild the spectrum."""
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design, photometry)
    coeffs = reg.coef_
    recon_spec = coeffs @ basis
    return coeffs, recon_spec

# --------------------------------------------------------------------------- #
# Demonstration on a single example
# --------------------------------------------------------------------------- #

sample_idx = 0
true_coeffs = coeff_matrix[sample_idx]
true_spectrum = spectra[sample_idx]
true_photometry = photometries[sample_idx]

recon_coeffs, recon_spectrum = reconstruct_spectrum(true_photometry)

print("True coefficients:", true_coeffs)
print("Reconstructed coefficients:", recon_coeffs)
print("\nTrue spectrum vs reconstructed spectrum:")
print("Mean absolute error:", np.mean(np.abs(true_spectrum - recon_spectrum)))