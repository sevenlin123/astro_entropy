#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# ----------------------------------------------------------------------
# 1. Define a spectral model (basis functions)
# ----------------------------------------------------------------------
def gaussian_basis(wavelengths, centers, widths):
    """
    Create a set of Gaussian basis functions.

    Parameters
    ----------
    wavelengths : ndarray
        Array of wavelength points.
    centers : list or ndarray
        Central wavelengths of the Gaussians.
    widths : list or ndarray
        Standard deviations of the Gaussians.

    Returns
    -------
    basis : ndarray
        Shape (n_basis, len(wavelengths)).
    """
    basis = []
    for c, w in zip(centers, widths):
        basis.append(norm.pdf(wavelengths, loc=c, scale=w))
    return np.vstack(basis)


# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis, noise_sigma=0.0, rng=None):
    """
    Generate synthetic spectra as linear combinations of basis functions.

    Parameters
    ----------
    n_samples : int
        Number of synthetic spectra to generate.
    basis : ndarray
        Basis matrix of shape (n_basis, n_wavelength).
    noise_sigma : float, optional
        Standard deviation of Gaussian noise added to the spectra.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    spectra : ndarray
        Shape (n_samples, n_wavelength).
    coeffs : ndarray
        Shape (n_samples, n_basis), true coefficients used.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_basis, n_wave = basis.shape
    coeffs = rng.normal(size=(n_samples, n_basis))
    spectra = coeffs @ basis
    if noise_sigma > 0.0:
        spectra += rng.normal(scale=noise_sigma, size=spectra.shape)
    return spectra, coeffs


# ----------------------------------------------------------------------
# 3. Define filter responses and generate photometric data
# ----------------------------------------------------------------------
def filter_response(wavelengths, center, width):
    """
    Gaussian filter transmission curve.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array.
    center : float
        Center wavelength of the filter.
    width : float
        Standard deviation of the filter.

    Returns
    -------
    response : ndarray
        Transmission values.
    """
    return norm.pdf(wavelengths, loc=center, scale=width)


def generate_photometry(spectra, wavelengths, filter_centers, filter_widths):
    """
    Compute photometric fluxes by integrating spectra over filter responses.

    Parameters
    ----------
    spectra : ndarray
        Shape (n_samples, n_wavelength).
    wavelengths : ndarray
        Wavelength grid.
    filter_centers : list or ndarray
        Centers of the filters.
    filter_widths : list or ndarray
        Widths of the filters.

    Returns
    -------
    photometry : ndarray
        Shape (n_samples, n_filters).
    """
    dw = np.diff(wavelengths)
    dw = np.concatenate([dw, [dw[-1]]])  # last interval
    responses = [filter_response(wavelengths, c, w) for c, w in zip(filter_centers, filter_widths)]
    photometry = []
    for resp in responses:
        flux = spectra * resp[:, None]  # broadcast
        photometry.append(np.sum(flux * dw, axis=1))
    return np.column_stack(photometry)


# ----------------------------------------------------------------------
# 4. Reconstruct spectra from photometry
# ----------------------------------------------------------------------
def reconstruct_spectra_from_photometry(photometry, wavelengths, filter_centers,
                                        filter_widths, basis, alpha=1e-6):
    """
    Reconstruct spectra using linear regression on basis coefficients.

    Parameters
    ----------
    photometry : ndarray
        Observed photometric fluxes (n_samples, n_filters).
    wavelengths : ndarray
        Wavelength grid.
    filter_centers : list or ndarray
        Filter centers.
    filter_widths : list or ndarray
        Filter widths.
    basis : ndarray
        Basis matrix (n_basis, n_wavelength).
    alpha : float, optional
        Regularization strength for ridge regression.

    Returns
    -------
    recon_spectra : ndarray
        Reconstructed spectra (n_samples, n_wavelength).
    """
    # Build design matrix mapping coefficients to photometry
    dw = np.diff(wavelengths)
    dw = np.concatenate([dw, [dw[-1]]])
    n_filters = len(filter_centers)
    design = np.zeros((n_filters, basis.shape[0]))
    for i, (c, w) in enumerate(zip(filter_centers, filter_widths)):
        resp = filter_response(wavelengths, c, w)
        design[i] = np.sum(basis * resp[:, None] * dw[:, None], axis=1)
    # Solve for coefficients using ridge regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design.T, photometry.T)
    coeffs_est = reg.predict(design.T).T
    # Reconstruct spectra
    recon_spectra = coeffs_est @ basis
    return recon_spectra


# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid (nm)
    wavelengths = np.linspace(300, 1000, 700)

    # Spectral basis (5 Gaussian features)
    basis_centers = [350, 450, 550, 650, 750]
    basis_widths = [20, 25, 30, 35, 40]
    basis = gaussian_basis(wavelengths, basis_centers, basis_widths)

    # Generate synthetic spectra
    n_samples = 10
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis, noise_sigma=0.01, rng=rng)

    # Define photometric filters (4 Gaussian filters)
    filter_centers = [400, 500, 600, 700]
    filter_widths = [30, 30, 30, 30]

    # Generate photometric data
    photometry = generate_photometry(spectra, wavelengths, filter_centers, filter_widths)

    # Reconstruct spectra from photometry
    recon_spectra = reconstruct_spectra_from_photometry(
        photometry, wavelengths, filter_centers, filter_widths, basis
    )

    # Compute reconstruction error
    mse = np.mean((spectra - recon_spectra) ** 2)
    print(f"Mean squared reconstruction error: {mse:.5f}")

    # Optional: inspect first spectrum and its reconstruction
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectra[0], label="True Spectrum")
    plt.plot(wavelengths, recon_spectra[0], '--', label="Reconstructed")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Spectrum Reconstruction Example")
    plt.tight_layout()
    plt.show()