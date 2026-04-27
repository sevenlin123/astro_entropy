#!/usr/bin/env python3
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

# ------------------------------
# Configuration parameters
# ------------------------------
N_SAMPLES   = 50          # number of synthetic stars
N_WAVES     = 200         # number of wavelength points in a spectrum
N_FILTERS   = 6           # number of photometric bands
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ------------------------------
# Generate wavelength grid
# ------------------------------
wavelengths = np.linspace(0.2, 1.0, N_WAVES)  # microns
delta_lam   = wavelengths[1] - wavelengths[0]

# ------------------------------
# Define a simple spectral model
# ------------------------------
def basis_function(wl, center, width):
    """Gaussian basis function."""
    return np.exp(-0.5 * ((wl - center) / width)**2)

# Build a set of basis functions (e.g., 10 Gaussians)
N_BASIS = 10
basis_centers = np.linspace(0.25, 0.95, N_BASIS)
basis_widths  = 0.05 * np.ones(N_BASIS)

BASIS_FUNCS = [basis_function(wavelengths, c, w) for c, w in zip(basis_centers, basis_widths)]
BASIS_FUNCS = np.vstack(BASIS_FUNCS)  # shape (N_BASIS, N_WAVES)

# ------------------------------
# Generate synthetic spectra
# ------------------------------
def generate_synthetic_spectra(n_samples, basis_funcs):
    """
    Generate synthetic spectra as linear combinations of basis functions
    with random coefficients and added Gaussian noise.
    """
    coeffs = np.random.randn(n_samples, basis_funcs.shape[0])
    spectra = coeffs @ basis_funcs  # linear combination
    spectra += 0.05 * np.random.randn(*spectra.shape)  # add noise
    return spectra, coeffs

# ------------------------------
# Define photometric bandpasses
# ------------------------------
def gaussian_bandpass(center, width, wl_grid):
    """Gaussian transmission curve."""
    return np.exp(-0.5 * ((wl_grid - center) / width)**2)

filter_centers = np.linspace(0.3, 0.9, N_FILTERS)
filter_widths  = 0.07 * np.ones(N_FILTERS)

FILTERS = np.vstack([gaussian_bandpass(c, w, wavelengths)
                     for c, w in zip(filter_centers, filter_widths)])
# Normalize each filter to unit area
FILTERS /= FILTERS.sum(axis=1, keepdims=True)

# ------------------------------
# Compute photometry from spectra
# ------------------------------
def compute_photometry(spectra, filters):
    """
    Integrate spectra over filter transmission curves.
    """
    # spectra: (n_samples, n_wavelengths)
    # filters: (n_filters, n_wavelengths)
    # Result: (n_samples, n_filters)
    return spectra @ filters.T * delta_lam

# ------------------------------
# Reconstruct spectra from photometry
# ------------------------------
def reconstruct_spectra(photometry, filters, n_basis=20):
    """
    Reconstruct spectra via linear regression on basis functions.
    Uses Ridge regression to regularize.
    """
    # Build design matrix: photometry = filters @ spectrum
    # We express spectrum as basis_coeffs @ BASIS_FUNCS
    # => photometry = filters @ (coeffs @ BASIS_FUNCS.T)
    # Let A = filters @ BASIS_FUNCS.T  -> shape (n_filters, n_basis)
    A = filters @ BASIS_FUNCS.T

    # Fit Ridge regression: photometry ~ A @ coeffs
    reg = Ridge(alpha=1.0, fit_intercept=False, solver='auto')
    reg.fit(A.T, photometry.T)  # transposed to shape (n_basis, n_samples)
    coeffs_est = reg.coef_.T  # shape (n_samples, n_basis)

    # Reconstruct spectra
    spectra_est = coeffs_est @ BASIS_FUNCS
    return spectra_est, coeffs_est

# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    # Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(N_SAMPLES, BASIS_FUNCS)

    # Compute photometric measurements
    photometry = compute_photometry(spectra_true, FILTERS)

    # Reconstruct spectra from photometry
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, FILTERS)

    # Simple evaluation
    mse = np.mean((spectra_true - spectra_rec)**2)
    print(f"Mean squared error between true and reconstructed spectra: {mse:.4e}")
    print(f"Shape of true spectra: {spectra_true.shape}")
    print(f"Shape of photometry: {photometry.shape}")
    print(f"Shape of reconstructed spectra: {spectra_rec.shape}")