#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

def create_wavelength_grid(wmin, wmax, n_points):
    """Uniformly spaced wavelength grid."""
    return np.linspace(wmin, wmax, n_points)

def build_gaussian_basis(n_basis, wavelength, rng=None):
    """
    Construct a set of Gaussian basis functions evaluated on `wavelength`.
    
    Parameters
    ----------
    n_basis : int
        Number of Gaussian basis functions.
    wavelength : ndarray
        1‑D array of wavelengths.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    basis : ndarray
        Shape (n_basis, len(wavelength))
    """
    if rng is None:
        rng = np.random.default_rng()
    wmin, wmax = wavelength[0], wavelength[-1]
    widths = rng.uniform(0.05*(wmax-wmin), 0.15*(wmax-wmin), size=n_basis)
    centers = rng.uniform(wmin, wmax, size=n_basis)
    basis = np.zeros((n_basis, len(wavelength)))
    for i in range(n_basis):
        basis[i] = norm.pdf(wavelength, loc=centers[i], scale=widths[i])
    return basis

def build_tophat_filters(n_filters, wavelength, rng=None):
    """
    Build simple top‑hat filter transmission curves.
    
    Parameters
    ----------
    n_filters : int
        Number of filters.
    wavelength : ndarray
        Wavelength grid.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    filters : ndarray
        Shape (n_filters, len(wavelength))
    """
    if rng is None:
        rng = np.random.default_rng()
    wmin, wmax = wavelength[0], wavelength[-1]
    widths = rng.uniform(0.1*(wmax-wmin), 0.25*(wmax-wmin), size=n_filters)
    centers = rng.uniform(wmin, wmax, size=n_filters)
    filters = np.zeros((n_filters, len(wavelength)))
    for j in range(n_filters):
        left = centers[j] - 0.5*widths[j]
        right = centers[j] + 0.5*widths[j]
        filters[j] = ((wavelength >= left) & (wavelength <= right)).astype(float)
    return filters

def synthesize_spectra(n_samples, basis, rng=None):
    """
    Produce synthetic spectra as random linear combinations of the basis.
    
    Parameters
    ----------
    n_samples : int
        Number of spectra to generate.
    basis : ndarray
        Shape (n_basis, n_wavelengths)
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    spectra : ndarray
        Shape (n_samples, n_wavelengths)
    coeffs : ndarray
        Shape (n_samples, n_basis)
    """
    if rng is None:
        rng = np.random.default_rng()
    n_basis = basis.shape[0]
    coeffs = rng.uniform(-1.0, 1.0, size=(n_samples, n_basis))
    spectra = coeffs @ basis
    return spectra, coeffs

def compute_photometry(spectra, filters):
    """
    Integrate spectra through filter transmission curves.
    
    Parameters
    ----------
    spectra : ndarray
        Shape (n_samples, n_wavelengths)
    filters : ndarray
        Shape (n_filters, n_wavelengths)
        
    Returns
    -------
    photometry : ndarray
        Shape (n_samples, n_filters)
    """
    return spectra @ filters.T

def reconstruct_coefficients(photometry, basis, filters):
    """
    Recover expansion coefficients from photometry.
    
    Parameters
    ----------
    photometry : ndarray
        Shape (n_samples, n_filters)
    basis : ndarray
        Shape (n_basis, n_wavelengths)
    filters : ndarray
        Shape (n_filters, n_wavelengths)
        
    Returns
    -------
    coeffs_rec : ndarray
        Shape (n_samples, n_basis)
    """
    # Design matrix: A_{i,j} = ∫ B_i(λ) T_j(λ) dλ
    A = basis @ filters.T           # shape (n_basis, n_filters)
    A_plus = np.linalg.pinv(A)      # shape (n_filters, n_basis)
    coeffs_rec = photometry @ A_plus
    return coeffs_rec

def reconstruct_spectrum(coeffs_rec, basis):
    """
    Rebuild spectra from recovered coefficients.
    
    Parameters
    ----------
    coeffs_rec : ndarray
        Shape (n_samples, n_basis)
    basis : ndarray
        Shape (n_basis, n_wavelengths)
        
    Returns
    -------
    spectra_rec : ndarray
        Shape (n_samples, n_wavelengths)
    """
    return coeffs_rec @ basis

def main():
    rng = np.random.default_rng(42)

    # Spectral domain
    wmin, wmax, n_wave = 4000., 8000., 1000
    wavelength = create_wavelength_grid(wmin, wmax, n_wave)

    # Build basis and filters
    n_basis = 10
    n_filters = 5
    basis = build_gaussian_basis(n_basis, wavelength, rng)
    filters = build_tophat_filters(n_filters, wavelength, rng)

    # Synthetic spectra
    n_samples = 50
    spectra, coeffs_true = synthesize_spectra(n_samples, basis, rng)

    # Photometry
    photometry = compute_photometry(spectra, filters)

    # Reconstruction
    coeffs_rec = reconstruct_coefficients(photometry, basis, filters)
    spectra_rec = reconstruct_spectrum(coeffs_rec, basis)

    # Simple error metric
    mse = np.mean((spectra - spectra_rec)**2)
    print(f"Reconstruction MSE: {mse:.4e}")

if __name__ == "__main__":
    main()