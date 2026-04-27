#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.
"""

import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# 1. Spectral model – a linear combination of Gaussian basis functions
# --------------------------------------------------------------------------- #

def gaussian_basis(wavelengths, n_basis=5, width=50.0):
    """
    Create a set of Gaussian basis functions evenly spaced over the wavelength grid.
    Parameters
    ----------
    wavelengths : ndarray
        1D array of wavelengths (nm).
    n_basis : int
        Number of Gaussian basis functions.
    width : float
        Width (sigma) of each Gaussian (nm).
    Returns
    -------
    basis : ndarray
        Shape (len(wavelengths), n_basis)
    """
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / width) ** 2)
    # Normalize each basis function to unit area
    basis /= basis.sum(axis=0, keepdims=True)
    return basis


def spectral_model(params, basis):
    """
    Linear combination of basis functions.
    Parameters
    ----------
    params : ndarray
        Coefficients for each basis function. Shape (n_basis,).
    basis : ndarray
        Basis functions. Shape (n_wavelengths, n_basis).
    Returns
    -------
    spectrum : ndarray
        Flux at each wavelength. Shape (n_wavelengths,).
    """
    return basis @ params


# --------------------------------------------------------------------------- #
# 2. Synthetic data generation
# --------------------------------------------------------------------------- #

def generate_wavelength_grid(start=400.0, stop=800.0, num=1000):
    """
    Generate a regular wavelength grid (nm).
    """
    return np.linspace(start, stop, num)


def generate_synthetic_spectra(n_spectra, basis, rng=np.random.default_rng()):
    """
    Generate synthetic spectra with random coefficients.
    Returns
    -------
    spectra : ndarray
        Shape (n_spectra, n_wavelengths)
    coeffs : ndarray
        Shape (n_spectra, n_basis)
    """
    n_basis = basis.shape[1]
    coeffs = rng.normal(loc=1.0, scale=0.5, size=(n_spectra, n_basis))
    spectra = coeffs @ basis.T  # (n_spectra, n_wavelengths)
    return spectra, coeffs


# --------------------------------------------------------------------------- #
# 3. Photometric simulation
# --------------------------------------------------------------------------- #

def generate_top_hat_filters(wavelengths, n_filters=4):
    """
    Create a few top‑hat filters covering the wavelength range.
    Returns
    -------
    filters : ndarray
        Shape (n_filters, n_wavelengths)
    """
    n_wl = len(wavelengths)
    bins = np.array_split(np.arange(n_wl), n_filters)
    filters = np.zeros((n_filters, n_wl))
    for i, idx in enumerate(bins):
        filters[i, idx] = 1.0
    # Normalize filters to equal width
    filters /= filters.sum(axis=1, keepdims=True)
    return filters


def simulate_photometry(spectra, filters, noise_std=0.02, rng=np.random.default_rng()):
    """
    Integrate spectra over filters to obtain photometric fluxes.
    Adds Gaussian noise.
    Parameters
    ----------
    spectra : ndarray
        Shape (n_spectra, n_wavelengths)
    filters : ndarray
        Shape (n_filters, n_wavelengths)
    noise_std : float
        Standard deviation of additive noise.
    Returns
    -------
    photometry : ndarray
        Shape (n_spectra, n_filters)
    """
    # Simple trapezoidal integration
    photometry = spectra @ filters.T  # (n_spectra, n_filters)
    noise = rng.normal(scale=noise_std, size=photometry.shape)
    return photometry + noise


# --------------------------------------------------------------------------- #
# 4. Reconstruction from photometry
# --------------------------------------------------------------------------- #

def reconstruct_from_photometry(photometry, filters, basis):
    """
    Reconstruct spectra from photometry by solving a linear inverse problem.
    Parameters
    ----------
    photometry : ndarray
        Observed fluxes. Shape (n_spectra, n_filters).
    filters : ndarray
        Filter transmission curves. Shape (n_filters, n_wavelengths).
    basis : ndarray
        Basis functions. Shape (n_wavelengths, n_basis).
    Returns
    -------
    reconstructed_spectra : ndarray
        Shape (n_spectra, n_wavelengths)
    coeffs_estimated : ndarray
        Estimated coefficients. Shape (n_spectra, n_basis)
    """
    # Build design matrix: M_{jk} = ∫ filter_j * basis_k
    M = filters @ basis  # (n_filters, n_basis)

    # Solve least-squares for each spectrum
    reg = LinearRegression(fit_intercept=False, positive=True)
    reg.fit(M, photometry.T)  # M: (n_filters, n_basis), target: (n_filters, n_spectra)
    coeffs_estimated = reg.coef_.T  # (n_spectra, n_basis)

    reconstructed_spectra = coeffs_estimated @ basis.T  # (n_spectra, n_wavelengths)
    return reconstructed_spectra, coeffs_estimated


# --------------------------------------------------------------------------- #
# 5. Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid
    wl = generate_wavelength_grid()

    # Basis functions
    basis = gaussian_basis(wl, n_basis=6, width=20.0)

    # Generate synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(10, basis, rng=rng)

    # Filters
    filt = generate_top_hat_filters(wl, n_filters=4)

    # Simulate photometry
    phot = simulate_photometry(spectra, filt, noise_std=0.01, rng=rng)

    # Reconstruction
    recon_spec, est_coeffs = reconstruct_from_photometry(phot, filt, basis)

    # Print comparison for first spectrum
    print("True coefficients:", true_coeffs[0])
    print("Estimated coefficients:", est_coeffs[0])
    print("\nFlux difference (norm) for first spectrum:",
          np.linalg.norm(spectra[0] - recon_spec[0]))