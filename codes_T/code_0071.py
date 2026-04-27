#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.
"""

import numpy as np
from sklearn.linear_model import Ridge


def create_wavelength_grid(start=400.0, stop=800.0, n_points=1000):
    """Create a regular wavelength grid."""
    return np.linspace(start, stop, n_points)


def gaussian(x, mu, sigma):
    """Gaussian function."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_basis(wl, n_basis=10, sigma=20.0):
    """
    Build a set of Gaussian basis functions on the wavelength grid.
    Returns a matrix of shape (n_wl, n_basis).
    """
    mus = np.linspace(wl[0] + sigma, wl[-1] - sigma, n_basis)
    basis = np.vstack([gaussian(wl, mu, sigma) for mu in mus]).T
    return basis


def generate_coefficients(n_samples, n_basis, std=1.0):
    """Generate random coefficients for synthetic spectra."""
    return np.random.randn(n_samples, n_basis) * std


def synthesize_spectra(basis, coeffs):
    """
    Construct synthetic spectra as linear combinations of basis functions.
    Returns an array of shape (n_samples, n_wl).
    """
    spectra = coeffs @ basis.T
    return spectra


def create_filters(n_filters=5, wl=None):
    """
    Create Gaussian filter responses.
    Returns an array of shape (n_filters, n_wl).
    """
    if wl is None:
        raise ValueError("Wavelength grid must be provided.")
    centers = np.linspace(wl[0] + 50, wl[-1] - 50, n_filters)
    width = 50.0
    filters = np.vstack([gaussian(wl, c, width) for c in centers])
    return filters


def photometry_from_spectra(spectra, filters, wl):
    """
    Integrate spectra over filter responses to produce photometric fluxes.
    spectra: (n_samples, n_wl)
    filters: (n_filters, n_wl)
    Returns an array of shape (n_samples, n_filters).
    """
    dlam = wl[1] - wl[0]
    return spectra @ filters.T * dlam


def compute_filter_matrix(filters, basis, wl):
    """
    Pre-compute the mapping from basis coefficients to photometric fluxes.
    Returns a matrix A of shape (n_filters, n_basis).
    """
    dlam = wl[1] - wl[0]
    return filters @ basis * dlam


def reconstruct_coefficients(y, A, alpha=1e-2):
    """
    Reconstruct basis coefficients from photometric measurements.
    y: (n_samples, n_filters)
    A: (n_filters, n_basis)
    Returns coeffs of shape (n_samples, n_basis).
    """
    # Closed‑form ridge solution
    inv = np.linalg.inv(A.T @ A + alpha * np.eye(A.shape[1]))
    coeffs = (inv @ A.T @ y.T).T
    return coeffs


def reconstruct_spectra(coeffs, basis):
    """
    Convert reconstructed coefficients back into spectra.
    coeffs: (n_samples, n_basis)
    Returns spectra of shape (n_samples, n_wl).
    """
    return coeffs @ basis.T


def main():
    # Set up wavelength grid and basis
    wl = create_wavelength_grid()
    basis = build_basis(wl, n_basis=10, sigma=20.0)

    # Generate synthetic spectra
    n_samples = 5
    coeff_true = generate_coefficients(n_samples, basis.shape[1], std=1.0)
    spectra_true = synthesize_spectra(basis, coeff_true)

    # Create filters and generate photometric data
    filters = create_filters(n_filters=5, wl=wl)
    photometry = photometry_from_spectra(spectra_true, filters, wl)

    # Reconstruct coefficients and spectra
    A = compute_filter_matrix(filters, basis, wl)
    coeff_est = reconstruct_coefficients(photometry, A, alpha=1e-2)
    spectra_rec = reconstruct_spectra(coeff_est, basis)

    # Evaluate reconstruction quality
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec) ** 2, axis=1))
    for i, e in enumerate(rmse):
        print(f"Sample {i} RMSE: {e:.4f}")


if __name__ == "__main__":
    main()