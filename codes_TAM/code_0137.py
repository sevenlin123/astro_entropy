#!/usr/bin/env python3
"""
Minimal reconstruction framework:
  1. Define spectral model (basis functions)
  2. Generate synthetic spectra
  3. Generate photometric data from synthetic spectra
  4. Reconstruct spectra from photometric measurements
"""

import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ---------- Utility functions ----------
def wavelength_grid(n_points=1000, wl_start=300, wl_stop=1000):
    """Generate a linear wavelength grid (in nm)."""
    return np.linspace(wl_start, wl_stop, n_points)


def gaussian_basis(wavelength, centers, width):
    """
    Build a matrix of Gaussian basis functions.
    Parameters
    ----------
    wavelength : ndarray
        Wavelength grid.
    centers : list or ndarray
        Centers of Gaussians.
    width : float
        Standard deviation of Gaussians.
    Returns
    -------
    basis : ndarray, shape (len(wavelength), len(centers))
    """
    basis = np.exp(-0.5 * ((wavelength[:, None] - centers[None, :]) / width)**2)
    # Normalize each basis vector
    basis /= np.sqrt(trapz(basis**2, wavelength, axis=0))
    return basis


def top_hat_filter(wavelength, wl_min, wl_max):
    """Return a top-hat transmission curve."""
    filt = np.zeros_like(wavelength)
    filt[(wavelength >= wl_min) & (wavelength <= wl_max)] = 1.0
    return filt


def filter_bank(wavelength, filter_edges):
    """
    Create a matrix of filter transmissions.
    Parameters
    ----------
    wavelength : ndarray
    filter_edges : list of tuples [(wl_min, wl_max), ...]
    Returns
    -------
    filters : ndarray, shape (len(wavelength), len(filter_edges))
    """
    return np.column_stack([top_hat_filter(wavelength, *edge) for edge in filter_edges])


def integrate_over_filter(spectrum, filt):
    """Compute integrated flux of a spectrum through a filter."""
    return trapz(spectrum * filt, x=None) / np.sum(filt)  # average flux


# ---------- Synthetic data generation ----------
def generate_synthetic_spectra(n_spectra, basis_matrix, coeff_dist=np.random.randn):
    """
    Generate synthetic spectra as linear combinations of basis functions.
    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate.
    basis_matrix : ndarray, shape (n_wavelength, n_basis)
    coeff_dist : callable
        Function to generate coefficients.
    Returns
    -------
    spectra : ndarray, shape (n_spectra, n_wavelength)
    coeffs : ndarray, shape (n_spectra, n_basis)
    """
    coeffs = coeff_dist(size=(n_spectra, basis_matrix.shape[1]))
    spectra = coeffs @ basis_matrix.T  # shape (n_spectra, n_wavelength)
    return spectra, coeffs


def generate_photometry(spectra, filters, noise_std=0.01):
    """
    Convert spectra to photometric measurements.
    Parameters
    ----------
    spectra : ndarray, shape (n_spectra, n_wavelength)
    filters : ndarray, shape (n_wavelength, n_filters)
    noise_std : float
        Standard deviation of Gaussian noise added to photometry.
    Returns
    -------
    photometry : ndarray, shape (n_spectra, n_filters)
    """
    n_spectra = spectra.shape[0]
    n_filters = filters.shape[1]
    photometry = np.empty((n_spectra, n_filters))
    for i in range(n_filters):
        photometry[:, i] = np.array([integrate_over_filter(s, filters[:, i])
                                     for s in spectra])
    # Add noise
    photometry += noise_std * np.random.randn(*photometry.shape)
    return photometry


# ---------- Reconstruction ----------
def build_design_matrix(filters, basis_matrix):
    """
    Construct the matrix that maps basis coefficients to photometric fluxes.
    Each element A_ij = integral_{λ} B_j(λ) * F_i(λ) dλ
    Parameters
    ----------
    filters : ndarray, shape (n_wavelength, n_filters)
    basis_matrix : ndarray, shape (n_wavelength, n_basis)
    Returns
    -------
    design : ndarray, shape (n_filters, n_basis)
    """
    # For each filter, compute integrals of each basis function
    design = np.array([
        [integrate_over_filter(basis_matrix[:, j], filters[:, i])
         for j in range(basis_matrix.shape[1])]
        for i in range(filters.shape[1])
    ])
    return design


def reconstruct_coefficients(photometry, design_matrix):
    """
    Fit linear regression model to recover basis coefficients.
    Parameters
    ----------
    photometry : ndarray, shape (n_spectra, n_filters)
    design_matrix : ndarray, shape (n_filters, n_basis)
    Returns
    -------
    coeffs_pred : ndarray, shape (n_spectra, n_basis)
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix.T, photometry.T)  # shape (n_filters, n_spectra)
    coeffs_pred = reg.coef_.T  # shape (n_spectra, n_basis)
    return coeffs_pred


def reconstruct_spectra(coeffs_pred, basis_matrix):
    """
    Compute reconstructed spectra from predicted coefficients.
    Parameters
    ----------
    coeffs_pred : ndarray, shape (n_spectra, n_basis)
    basis_matrix : ndarray, shape (n_wavelength, n_basis)
    Returns
    -------
    spectra_rec : ndarray, shape (n_spectra, n_wavelength)
    """
    return coeffs_pred @ basis_matrix.T


# ---------- Main ----------
def main():
    np.random.seed(42)

    # 1. Define wavelength grid and basis functions
    wav = wavelength_grid()
    centers = np.linspace(350, 950, 10)  # 10 Gaussian bases
    basis = gaussian_basis(wav, centers, width=30)

    # 2. Define filter bank (5 top-hat filters)
    filter_edges = [(300, 400), (400, 500), (500, 600),
                    (600, 700), (700, 800)]
    filters = filter_bank(wav, filter_edges)

    # 3. Generate synthetic spectra
    n_samples = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_samples, basis, coeff_dist=lambda size: np.random.randn(*size))

    # 4. Generate photometric data
    photometry = generate_photometry(spectra_true, filters, noise_std=0.02)

    # 5. Build design matrix for reconstruction
    design = build_design_matrix(filters, basis)

    # 6. Reconstruct coefficients from photometry
    coeffs_est = reconstruct_coefficients(photometry, design)

    # 7. Reconstruct spectra
    spectra_rec = reconstruct_spectra(coeffs_est, basis)

    # Print summary statistics
    print("True vs. Estimated Coefficients (first 5 samples):")
    print(np.round(np.hstack([coeffs_true[:5], coeffs_est[:5]]), 3))
    print("\nReconstruction error (RMSE per sample):")
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec)**2, axis=1))
    print(np.round(rmse, 3))


if __name__ == "__main__":
    main()