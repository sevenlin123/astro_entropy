#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import Ridge

# ---------- Spectral model ----------
def gaussian_basis(n_basis, wl):
    """
    Create n_basis Gaussian basis functions over wavelengths wl.
    Returns a matrix of shape (len(wl), n_basis).
    """
    centers = np.linspace(wl[0]+0.1*(wl[-1]-wl[0]),
                          wl[-1]-0.1*(wl[-1]-wl[0]), n_basis)
    widths = 0.05 * (wl[-1]-wl[0]) / n_basis * np.ones_like(centers)
    B = np.exp(-0.5 * ((wl[:, None] - centers)**2) / widths**2)
    return B

# ---------- Synthetic data generation ----------
def generate_synthetic_spectra(n_samples, B):
    """
    Generate synthetic spectra as random linear combinations of basis B.
    Returns:
        coeffs: (n_samples, n_basis)
        spectra: (n_samples, n_wavelength)
    """
    rng = np.random.default_rng(seed=42)
    coeffs = rng.normal(scale=1.0, size=(n_samples, B.shape[1]))
    spectra = coeffs @ B.T
    return coeffs, spectra

# ---------- Photometry ----------
def define_filters(wl):
    """
    Define a set of rectangular filters.
    Returns a list of transmission arrays, each of shape (len(wl),).
    """
    filter_ranges = [(400, 500), (500, 600), (600, 700), (700, 800)]
    filters = []
    for lo, hi in filter_ranges:
        trans = np.where((wl >= lo) & (wl <= hi), 1.0, 0.0)
        filters.append(trans)
    return filters

def photometry_from_spectra(spectra, filters, wl):
    """
    Integrate each spectrum over each filter.
    Returns a matrix of shape (n_samples, n_filters).
    """
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    phot = np.zeros((n_samples, n_filters))
    for i, filt in enumerate(filters):
        phot[:, i] = np.trapz(spectra * filt, wl, axis=1)
    return phot

# ---------- Reconstruction ----------
def build_measurement_matrix(B, filters, wl):
    """
    Build the matrix M = ∫ B_j * F_i dλ for all filters i and basis j.
    Shape: (n_filters, n_basis)
    """
    n_filters = len(filters)
    n_basis = B.shape[1]
    M = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            M[i, j] = np.trapz(B[:, j] * filt, wl)
    return M

def reconstruct_spectra(phot, M, n_basis):
    """
    Recover basis coefficients from photometry using Ridge regression.
    Returns estimated coefficients and reconstructed spectra.
    """
    reg = Ridge(alpha=1e-3, fit_intercept=False)
    reg.fit(phot, np.eye(phot.shape[0]))
    coeffs_hat = reg.predict(phot)
    # The above is not correct; we need to solve M * c = phot
    # Instead use least squares directly
    coeffs_hat, *_ = np.linalg.lstsq(M.T, phot.T, rcond=None)
    coeffs_hat = coeffs_hat.T
    return coeffs_hat

# ---------- Main execution ----------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(400, 800, 1000)          # nm

    # Build basis functions
    n_basis = 5
    B = gaussian_basis(n_basis, wl)

    # Generate synthetic spectra
    n_samples = 20
    true_coeffs, spectra = generate_synthetic_spectra(n_samples, B)

    # Define filters
    filters = define_filters(wl)

    # Compute photometric observations
    phot = photometry_from_spectra(spectra, filters, wl)

    # Build measurement matrix
    M = build_measurement_matrix(B, filters, wl)

    # Reconstruct spectra from photometry
    coeffs_recon = np.linalg.lstsq(M.T, phot.T, rcond=None)[0].T
    spectra_recon = coeffs_recon @ B.T

    # Evaluate reconstruction
    err = np.mean(np.abs(spectra - spectra_recon))
    print(f"Mean absolute reconstruction error per wavelength: {err:.3e}")