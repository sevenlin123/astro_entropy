#!/usr/bin/env python3
import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------
# 1. Spectral model: create a set of basis spectra
# ----------------------------------------------------------
def create_basis_spectra(n_basis, wave):
    """
    Generate n_basis synthetic spectra composed of Gaussian absorption features.
    Parameters
    ----------
    n_basis : int
        Number of basis spectra.
    wave : ndarray
        Wavelength grid (nm).
    Returns
    -------
    spectra : ndarray, shape (n_basis, len(wave))
        Basis spectra.
    """
    spectra = np.zeros((n_basis, len(wave)))
    np.random.seed(42)
    for i in range(n_basis):
        # Randomly place a few Gaussian absorption lines
        n_lines = np.random.randint(3, 7)
        line_centers = np.random.uniform(wave.min(), wave.max(), n_lines)
        line_widths = np.random.uniform(2.0, 5.0, n_lines)
        depths = np.random.uniform(0.1, 0.4, n_lines)
        spec = np.ones_like(wave)
        for c, w, d in zip(line_centers, line_widths, depths):
            spec -= d * np.exp(-(wave - c)**2 / (2 * w**2))
        spectra[i] = spec
    return spectra


# ----------------------------------------------------------
# 2. Generate synthetic spectra from random coefficients
# ----------------------------------------------------------
def generate_synthetic_spectrum(basis_spectra, coeffs, noise_std=0.01):
    """
    Combine basis spectra with given coefficients to produce a synthetic spectrum.
    Parameters
    ----------
    basis_spectra : ndarray, shape (n_basis, len(wave))
    coeffs : ndarray, shape (n_basis,)
    noise_std : float
        Standard deviation of additive Gaussian noise.
    Returns
    -------
    spectrum : ndarray
        Synthetic spectrum.
    """
    spectrum = np.dot(coeffs, basis_spectra)
    spectrum += np.random.normal(scale=noise_std, size=spectrum.shape)
    return spectrum


# ----------------------------------------------------------
# 3. Define filter transmission curves
# ----------------------------------------------------------
def create_filters(n_filters, wave):
    """
    Create simple Gaussian filter transmission curves.
    Parameters
    ----------
    n_filters : int
        Number of filters.
    wave : ndarray
        Wavelength grid.
    Returns
    -------
    filters : list of tuples
        Each tuple contains (center_wavelength, width, transmission_curve).
    """
    np.random.seed(24)
    filters = []
    for _ in range(n_filters):
        center = np.random.uniform(wave.min(), wave.max())
        width = np.random.uniform(30.0, 70.0)
        trans = np.exp(-0.5 * ((wave - center)/width)**2)
        filters.append((center, width, trans))
    return filters


# ----------------------------------------------------------
# 4. Compute photometric fluxes through filters
# ----------------------------------------------------------
def compute_photometry(spectrum, wave, filters):
    """
    Integrate spectrum over each filter transmission curve.
    Parameters
    ----------
    spectrum : ndarray
        Spectral flux density.
    wave : ndarray
        Wavelength grid.
    filters : list of tuples
        Filter definitions.
    Returns
    -------
    fluxes : ndarray, shape (len(filters),)
    """
    fluxes = np.zeros(len(filters))
    for i, (_, _, trans) in enumerate(filters):
        flux = trapz(spectrum * trans, wave) / trapz(trans, wave)
        fluxes[i] = flux
    return fluxes


# ----------------------------------------------------------
# 5. Reconstruct spectrum from photometry
# ----------------------------------------------------------
def reconstruct_spectrum_from_photometry(
        photometry, wave, filters, basis_spectra):
    """
    Fit linear combination of basis spectra to match photometric fluxes.
    Parameters
    ----------
    photometry : ndarray, shape (n_filters,)
        Observed photometric fluxes.
    wave : ndarray
        Wavelength grid.
    filters : list of tuples
        Filter definitions.
    basis_spectra : ndarray, shape (n_basis, len(wave))
        Basis spectra.
    Returns
    -------
    coeffs : ndarray, shape (n_basis,)
        Reconstructed coefficients.
    reconstructed_spectrum : ndarray
        Reconstructed spectral flux density.
    """
    # Build design matrix: integral of each basis spectrum through each filter
    n_basis = basis_spectra.shape[0]
    X = np.zeros((len(filters), n_basis))
    for j, (_, _, trans) in enumerate(filters):
        for k in range(n_basis):
            X[j, k] = trapz(basis_spectra[k] * trans, wave) / trapz(trans, wave)

    # Solve least-squares problem
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, photometry)
    coeffs = lr.coef_
    reconstructed_spectrum = np.dot(coeffs, basis_spectra)
    return coeffs, reconstructed_spectrum


# ----------------------------------------------------------
# Main execution: generate data and perform reconstruction
# ----------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (nm)
    wave = np.linspace(400, 800, 1000)

    # Create basis spectra
    n_basis = 6
    basis = create_basis_spectra(n_basis, wave)

    # Generate true coefficients
    np.random.seed(99)
    true_coeffs = np.random.uniform(0.5, 1.5, size=n_basis)

    # Generate synthetic spectrum
    true_spectrum = generate_synthetic_spectrum(basis, true_coeffs, noise_std=0.005)

    # Create filters
    n_filters = 4
    filters = create_filters(n_filters, wave)

    # Compute photometric measurements
    photometric_fluxes = compute_photometry(true_spectrum, wave, filters)

    # Reconstruct spectrum from photometry
    recon_coeffs, recon_spectrum = reconstruct_spectrum_from_photometry(
        photometric_fluxes, wave, filters, basis
    )

    # Display results
    print("True coefficients:\n", true_coeffs)
    print("\nRecovered coefficients:\n", recon_coeffs)

    # Plotting is omitted as per constraints