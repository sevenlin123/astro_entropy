#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression


def create_spectral_basis(n_wavelengths: int, n_components: int, wl_min=400, wl_max=800):
    """
    Create a simple spectral basis using Gaussian functions.
    Returns:
        wavelengths: 1-D array of wavelength points.
        basis: 2-D array (n_wavelengths, n_components).
    """
    wavelengths = np.linspace(wl_min, wl_max, n_wavelengths)
    centers = np.linspace(wl_min + 0.15 * (wl_max - wl_min),
                          wl_max - 0.15 * (wl_max - wl_min),
                          n_components)
    widths = (wl_max - wl_min) / (2 * n_components)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths)**2)
    # Normalize basis vectors to unit L2 norm
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    return wavelengths, basis


def generate_synthetic_spectra(n_spectra: int, basis: np.ndarray):
    """
    Generate synthetic spectra as linear combinations of the basis.
    Returns:
        spectra: 2-D array (n_spectra, n_wavelengths)
        coeffs: 2-D array (n_spectra, n_components)
    """
    n_components = basis.shape[1]
    coeffs = np.random.randn(n_spectra, n_components)
    spectra = coeffs @ basis.T
    return spectra, coeffs


def define_filters(n_filters: int, wl_min=400, wl_max=800):
    """
    Define simple top-hat filter transmission curves.
    Returns:
        filter_transmission: 2-D array (n_filters, n_wavelengths)
        filter_centers: array of filter central wavelengths
        filter_widths: array of filter widths
    """
    wavelengths = np.linspace(wl_min, wl_max, 500)
    centers = np.linspace(wl_min + 0.1 * (wl_max - wl_min),
                          wl_max - 0.1 * (wl_max - wl_min),
                          n_filters)
    widths = (wl_max - wl_min) / (2 * n_filters)
    filt_tr = np.zeros((n_filters, len(wavelengths)))
    for i, (c, w) in enumerate(zip(centers, widths)):
        filt_tr[i] = np.where(np.abs(wavelengths - c) < w / 2, 1.0, 0.0)
    return wavelengths, filt_tr, centers, widths


def compute_photometry(spectra: np.ndarray, filter_trans: np.ndarray, wavelengths: np.ndarray):
    """
    Compute photometric fluxes by integrating spectra with filter transmissions.
    Returns:
        photometry: 2-D array (n_spectra, n_filters)
    """
    # Use trapezoidal integration
    fluxes = []
    for filt in filter_trans:
        integrated = simps(spectra * filt, x=wavelengths, axis=1)
        fluxes.append(integrated)
    return np.column_stack(fluxes)


def reconstruct_spectra(photometry: np.ndarray,
                        basis: np.ndarray,
                        filter_trans: np.ndarray,
                        wavelengths: np.ndarray):
    """
    Reconstruct spectra from photometric data by solving for basis coefficients.
    Returns:
        reconstructed_spectra: 2-D array (n_spectra, n_wavelengths)
        recovered_coeffs: 2-D array (n_spectra, n_components)
    """
    # Build response matrix: integral of basis functions through each filter
    n_filters = filter_trans.shape[0]
    n_components = basis.shape[1]
    response = np.zeros((n_filters, n_components))
    for i in range(n_filters):
        for j in range(n_components):
            response[i, j] = simps(basis[:, j] * filter_trans[i], x=wavelengths)

    # Solve least-squares: photometry = response * coeffs.T
    reg = LinearRegression(fit_intercept=False)
    reg.fit(response, photometry.T)
    coeffs = reg.coef_.T
    reconstructed = coeffs @ basis.T
    return reconstructed, coeffs


def main():
    # Parameters
    n_wavelengths = 1000
    n_components = 10
    n_spectra = 20
    n_filters = 5

    # Generate basis and synthetic spectra
    wavelengths, basis = create_spectral_basis(n_wavelengths, n_components)
    spectra, true_coeffs = generate_synthetic_spectra(n_spectra, basis)

    # Define filters
    filt_wl, filt_tr, filt_centers, filt_widths = define_filters(n_filters)

    # Interpolate basis onto filter wavelengths
    # (basis already on wavelengths; assume same grid for simplicity)
    if not np.array_equal(wavelengths, filt_wl):
        raise RuntimeError("Wavelength grids must match for this demo.")

    # Compute photometry
    photometry = compute_photometry(spectra, filt_tr, wavelengths)

    # Reconstruct spectra
    recon_spectra, rec_coeffs = reconstruct_spectra(photometry, basis, filt_tr, wavelengths)

    # Evaluate reconstruction error
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared reconstruction error: {mse:.6f}")

    # Optional: compare true vs recovered coefficients
    coef_mse = np.mean((true_coeffs - rec_coeffs)**2)
    print(f"Mean squared coefficient error: {coef_mse:.6f}")


if __name__ == "__main__":
    main()