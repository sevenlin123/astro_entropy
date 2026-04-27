#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# 1. Spectral model
# ----------------------------------------------------------------------
def create_basis_spectra(n_bases, wavelengths):
    """Generate orthogonal basis spectra (e.g., sine waves)."""
    basis = []
    for k in range(1, n_bases + 1):
        spec = np.sin(k * np.pi * wavelengths / wavelengths[-1])
        basis.append(spec)
    return np.vstack(basis)  # shape (n_bases, n_wave)

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def synthesize_spectra(n_spectra, basis, rng=None):
    """Create synthetic spectra as random combinations of basis."""
    rng = np.random.default_rng(rng)
    coeffs = rng.normal(size=(n_spectra, basis.shape[0]))
    spectra = coeffs @ basis  # shape (n_spectra, n_wave)
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Generate photometric data
# ----------------------------------------------------------------------
def gaussian_filter(wavelengths, center, width):
    """Simple Gaussian transmission curve."""
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def create_filters(n_filters, wavelengths):
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_filters)
    widths = 0.05 * (wavelengths[-1] - wavelengths[0]) * np.ones(n_filters)
    filters = np.array([gaussian_filter(wavelengths, c, w)
                        for c, w in zip(centers, widths)])  # shape (n_filters, n_wave)
    return filters, centers

def photometry_from_spectra(spectra, filters):
    """Compute photometric fluxes: integral(s * f) / integral(f)."""
    integrals = simps(spectra[:, :, None] * filters[None, :, :], axis=2)
    norm = simps(filters, axis=1)[None, :]
    return integrals / norm  # shape (n_spectra, n_filters)

# ----------------------------------------------------------------------
# 4. Reconstruct spectrum from photometry
# ----------------------------------------------------------------------
def construct_design_matrix(filters, basis, wavelengths):
    """
    Build matrix mapping coefficients to photometric fluxes:
    For each filter, integrate basis spectra weighted by filter response.
    """
    n_filters, n_wave = filters.shape
    n_bases = basis.shape[0]
    A = np.zeros((n_filters, n_bases))
    for i in range(n_filters):
        for j in range(n_bases):
            A[i, j] = simps(basis[j] * filters[i], wavelengths) / simps(filters[i], wavelengths)
    return A  # shape (n_filters, n_bases)

def reconstruct_spectra(photometry, filters, basis, wavelengths, alpha=1e-3):
    """
    Reconstruct spectra by solving a regularised linear system:
    coeffs = (A^T A + alpha I)^-1 A^T photometry
    """
    A = construct_design_matrix(filters, basis, wavelengths)
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(A, photometry.T)          # fit per coefficient
    coeffs_rec = reg.coef_.T           # shape (n_spectra, n_bases)
    spectra_rec = coeffs_rec @ basis   # shape (n_spectra, n_wave)
    return spectra_rec, coeffs_rec

# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (e.g., 400–800 nm)
    wav = np.linspace(400, 800, 400)

    # Basis spectra
    n_bases = 5
    basis_spectra = create_basis_spectra(n_bases, wav)

    # Synthetic spectra
    n_spectra = 10
    spectra_true, coeffs_true = synthesize_spectra(n_spectra, basis_spectra, rng=42)

    # Filters
    n_filters = 7
    filters, centers = create_filters(n_filters, wav)

    # Photometric measurements
    phot = photometry_from_spectra(spectra_true, filters)

    # Reconstruction
    spectra_rec, coeffs_rec = reconstruct_spectra(phot, filters, basis_spectra, wav)

    # Compare (simple L2 error)
    err = np.linalg.norm(spectra_true - spectra_rec, axis=1) / np.linalg.norm(spectra_true, axis=1)
    print("Relative reconstruction errors per spectrum:", err)