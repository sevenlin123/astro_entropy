#!/usr/bin/env python3
"""
Minimal spectral reconstruction example:
    1. Define a spectral model using Gaussian basis functions.
    2. Generate synthetic spectra (random coefficients).
    3. Compute synthetic photometric measurements via filter integrations.
    4. Reconstruct spectra from photometry using ridge regression.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from sklearn.linear_model import Ridge


def gaussian_basis(wavelengths, centers, widths):
    """
    Build Gaussian basis matrix: shape (N_wavelengths, N_basis)
    """
    return np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :]) ** 2)


def model_spectrum(wavelengths, coeffs, centers, widths):
    """Generate spectrum from coefficients."""
    basis = gaussian_basis(wavelengths, centers, widths)
    return basis @ coeffs


def generate_filters(n_filters, wavelength_min, wavelength_max):
    """
    Create synthetic Gaussian filters:
        centers evenly spaced in log-wavelength space
        widths constant relative to center
    Returns list of (center, width) tuples.
    """
    centers = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_filters)
    widths = 0.05 * centers          # 5% fractional width
    return list(zip(centers, widths))


def filter_response(wavelengths, center, width):
    """Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def generate_synthetic_spectra(
    n_spectra,
    wavelengths,
    centers,
    widths,
    coeff_range=(-1, 1),
):
    """Randomly sample spectra coefficients and produce spectra."""
    coeffs = np.random.uniform(coeff_range[0], coeff_range[1], size=(n_spectra, len(centers)))
    spectra = np.array([model_spectrum(wavelengths, c, centers, widths) for c in coeffs])
    return spectra, coeffs


def compute_photometry(spectra, wavelengths, filters):
    """
    Integrate each spectrum over each filter response to get synthetic photometry.
    Returns array of shape (n_spectra, n_filters).
    """
    n_filters = len(filters)
    photometry = np.zeros((spectra.shape[0], n_filters))
    for i, (center, width) in enumerate(filters):
        trans = filter_response(wavelengths, center, width)
        # normalize by integral of transmission to mimic magnitude zero point
        norm = trapz(trans, wavelengths)
        photometry[:, i] = trapz(spectra * trans, wavelengths) / norm
    return photometry


def reconstruct_spectra(
    photometry,
    wavelengths,
    filters,
    centers,
    widths,
    alpha=1.0,
):
    """
    Estimate spectrum coefficients from photometry using ridge regression.
    Returns reconstructed spectra and estimated coefficients.
    """
    # Build design matrix A where A_ij = integral of basis_j * filter_i
    n_filters = len(filters)
    n_basis = len(centers)
    A = np.zeros((n_filters, n_basis))
    for i, (c_f, w_f) in enumerate(filters):
        trans = filter_response(wavelengths, c_f, w_f)
        for j, (c_b, w_b) in enumerate(zip(centers, widths)):
            basis = gaussian_basis(wavelengths, [c_b], [w_b])[:, 0]
            A[i, j] = trapz(basis * trans, wavelengths)
    # Solve ridge regression: coeffs = (A^T A + alpha I)^-1 A^T photometry.T
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    ridge.fit(A, photometry)
    coeffs_est = ridge.coef_
    # Reconstruct spectra from estimated coeffs
    spectra_est = np.array(
        [model_spectrum(wavelengths, c, centers, widths) for c in coeffs_est]
    )
    return spectra_est, coeffs_est.T


def main():
    # Parameters
    wavelength_min, wavelength_max = 400.0, 800.0   # nm
    n_points = 1000
    wavelengths = np.linspace(wavelength_min, wavelength_max, n_points)

    # Spectral basis
    n_basis = 8
    centers = np.linspace(wavelength_min, wavelength_max, n_basis)
    widths = 0.03 * centers  # 3% width

    # Filters
    n_filters = 5
    filters = generate_filters(n_filters, wavelength_min, wavelength_max)

    # Generate synthetic data
    n_spectra = 20
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_spectra, wavelengths, centers, widths
    )

    # Compute photometric measurements
    photometry = compute_photometry(spectra_true, wavelengths, filters)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(
        photometry, wavelengths, filters, centers, widths, alpha=0.1
    )

    # Simple error metrics
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared reconstruction error: {mse:.6f}")


if __name__ == "__main__":
    main()