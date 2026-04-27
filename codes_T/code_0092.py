#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.

Author: ChatGPT
"""

import numpy as np
from sklearn.linear_model import Ridge


def gaussian_basis(wavelengths: np.ndarray,
                   centers: np.ndarray,
                   sigma: float) -> np.ndarray:
    """
    Build a Gaussian basis set.

    Parameters
    ----------
    wavelengths : ndarray
        Array of wavelength values (1‑D).
    centers : ndarray
        Center wavelengths of the Gaussian components (1‑D).
    sigma : float
        Standard deviation of all Gaussian components.

    Returns
    -------
    basis : ndarray
        Matrix of shape (n_components, n_wavelengths).
    """
    return np.exp(-((wavelengths[:, None] - centers[None, :]) ** 2)
                  / (2 * sigma ** 2))


def generate_synthetic_spectra(n_spectra: int,
                               wavelengths: np.ndarray,
                               basis: np.ndarray,
                               amplitude_range: tuple = (0.5, 1.5)) -> tuple:
    """
    Generate synthetic spectra as random linear combinations of basis functions.

    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate.
    wavelengths : ndarray
        Wavelength grid.
    basis : ndarray
        Basis matrix (n_components, n_wavelengths).
    amplitude_range : tuple
        Range of random amplitudes.

    Returns
    -------
    spectra : ndarray
        Array of shape (n_spectra, n_wavelengths).
    coeffs : ndarray
        True coefficients used to generate each spectrum (n_spectra, n_components).
    """
    n_components = basis.shape[0]
    coeffs = np.random.uniform(amplitude_range[0],
                               amplitude_range[1],
                               size=(n_spectra, n_components))
    spectra = coeffs @ basis
    return spectra, coeffs


def create_filters(wavelengths: np.ndarray,
                   n_filters: int = 4) -> np.ndarray:
    """
    Create a set of simple top‑hat filter transmissions.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength grid.
    n_filters : int
        Number of filters to construct.

    Returns
    -------
    filters : ndarray
        Transmission matrix of shape (n_filters, n_wavelengths).
    """
    band_limits = [(400, 500), (520, 600), (620, 680), (690, 740)]
    filters = []
    for low, high in band_limits[:n_filters]:
        trans = np.where((wavelengths >= low) & (wavelengths <= high), 1.0, 0.0)
        filters.append(trans)
    return np.array(filters)


def photometry_from_spectra(spectra: np.ndarray,
                            filters: np.ndarray,
                            wavelengths: np.ndarray) -> np.ndarray:
    """
    Compute synthetic photometry by integrating spectra over filter transmissions.

    Parameters
    ----------
    spectra : ndarray
        Spectra array (n_spectra, n_wavelengths).
    filters : ndarray
        Filter transmissions (n_filters, n_wavelengths).
    wavelengths : ndarray
        Wavelength grid.

    Returns
    -------
    photometry : ndarray
        Integrated fluxes (n_spectra, n_filters).
    """
    dlam = wavelengths[1] - wavelengths[0]
    photometry = np.dot(spectra, filters.T) * dlam
    return photometry


def reconstruct_spectra_from_photometry(photometry: np.ndarray,
                                        filters: np.ndarray,
                                        wavelengths: np.ndarray,
                                        basis: np.ndarray,
                                        alpha: float = 1.0) -> tuple:
    """
    Reconstruct spectra from photometry using linear regression on the basis.

    Parameters
    ----------
    photometry : ndarray
        Observed photometric fluxes (n_spectra, n_filters).
    filters : ndarray
        Filter transmissions (n_filters, n_wavelengths).
    wavelengths : ndarray
        Wavelength grid.
    basis : ndarray
        Basis functions (n_components, n_wavelengths).
    alpha : float
        Regularization strength for Ridge regression.

    Returns
    -------
    reconstructed_spectra : ndarray
        Estimated spectra (n_spectra, n_wavelengths).
    estimated_coeffs : ndarray
        Estimated coefficients (n_spectra, n_components).
    """
    # Build the design matrix M (n_filters, n_components)
    dlam = wavelengths[1] - wavelengths[0]
    M = np.trapz(basis * filters[:, :, None], wavelengths, axis=2)
    # Fit Ridge regression for each spectrum
    n_spectra = photometry.shape[0]
    estimated_coeffs = np.zeros((n_spectra, basis.shape[0]))
    for i in range(n_spectra):
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(M, photometry[i])
        estimated_coeffs[i] = ridge.coef_
    reconstructed_spectra = estimated_coeffs @ basis
    return reconstructed_spectra, estimated_coeffs


def main() -> None:
    # Define wavelength grid and basis
    wavelengths = np.linspace(380.0, 750.0, 200)
    component_centers = np.linspace(410.0, 700.0, 5)
    sigma = 15.0
    basis = gaussian_basis(wavelengths, component_centers, sigma)

    # Generate synthetic spectra
    n_spectra = 10
    spectra, true_coeffs = generate_synthetic_spectra(
        n_spectra, wavelengths, basis
    )

    # Construct filter set
    filters = create_filters(wavelengths, n_filters=4)

    # Compute synthetic photometry
    photometry = photometry_from_spectra(spectra, filters, wavelengths)

    # Reconstruct spectra from photometry
    recon_spectra, est_coeffs = reconstruct_spectra_from_photometry(
        photometry, filters, wavelengths, basis, alpha=0.1
    )

    # Simple diagnostics
    mse = np.mean((spectra - recon_spectra) ** 2)
    print(f"Mean squared reconstruction error: {mse:.4e}")
    print("\nFirst three true vs estimated coefficients:")
    for i in range(3):
        print(f"Spectrum {i+1} true: {true_coeffs[i]}")
        print(f"             est.: {est_coeffs[i]}")


if __name__ == "__main__":
    main()