#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# 1. Spectral model ---------------------------------------------------
def build_basis(wavelength, n_basis):
    """Construct a set of Gaussian basis functions."""
    centers = np.linspace(wavelength[0], wavelength[-1], n_basis)
    widths  = (wavelength[-1] - wavelength[0]) / (2 * n_basis)
    basis = np.vstack([norm.pdf(wavelength, loc=c, scale=widths)
                       for c in centers]).T  # shape (N_wave, n_basis)
    return basis

# ------------------------------------------------------------------
# 2. Synthetic spectra -----------------------------------------------
def generate_synthetic_spectra(n_samples, basis, rng=np.random.default_rng()):
    """Generate spectra as random linear combinations of basis functions."""
    n_basis = basis.shape[1]
    coeffs = rng.normal(size=(n_samples, n_basis))
    spectra = coeffs @ basis.T  # shape (n_samples, N_wave)
    return spectra, coeffs

# ------------------------------------------------------------------
# 3. Photometric data generation -------------------------------------
def build_filters(wavelength, n_filters, rng=np.random.default_rng()):
    """Create random Gaussian filters."""
    centers = rng.uniform(wavelength[0], wavelength[-1], size=n_filters)
    widths  = (wavelength[-1] - wavelength[0]) / (4 * n_filters)
    filters = np.vstack([norm.pdf(wavelength, loc=c, scale=widths)
                         for c in centers]).T  # shape (n_filters, N_wave)
    return filters

def generate_photometry(spectra, filters):
    """Integrate spectra over filter transmission curves."""
    return spectra @ filters.T  # shape (n_samples, n_filters)

# ------------------------------------------------------------------
# 4. Reconstruction -----------------------------------------------
def reconstruct_spectra(photometry, filters, basis, alpha=1e-3):
    """
    Recover spectra from photometry by solving a linear inverse problem.
    Returns reconstructed spectra and estimated coefficients.
    """
    # Build mapping matrix A such that phot = A @ coeffs
    A = filters @ basis.T                 # shape (n_filters, n_basis)
    n_samples = photometry.shape[0]
    n_basis   = basis.shape[1]

    # Fit coefficients for each sample
    coeffs_est = np.zeros((n_samples, n_basis))
    for i in range(n_samples):
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(A, photometry[i])
        coeffs_est[i] = ridge.coef_

    # Reconstruct spectra
    spectra_rec = coeffs_est @ basis.T    # shape (n_samples, N_wave)
    return spectra_rec, coeffs_est

# ------------------------------------------------------------------
# 5. Demo ------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)

    # Wavelength grid
    wavelength = np.linspace(400, 700, 300)  # nm

    # Build spectral basis
    n_basis = 10
    basis = build_basis(wavelength, n_basis)

    # Generate synthetic spectra
    n_samples = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis, rng)

    # Build filters
    n_filters = 5
    filters = build_filters(wavelength, n_filters, rng)

    # Generate photometric measurements
    photometry = generate_photometry(spectra_true, filters)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, filters, basis)

    # Evaluate reconstruction
    err = np.linalg.norm(spectra_true - spectra_rec, axis=1)
    print(f"Mean reconstruction error (per sample): {err.mean():.3f}")

    # Show a comparison for the first sample
    idx = 0
    print("\nWavelength (nm)")
    print(wavelength[:5], '...', wavelength[-5:])
    print("\nTrue spectrum (first 5 points)")
    print(spectra_true[idx, :5], '...', spectra_true[idx, -5:])
    print("\nReconstructed spectrum (first 5 points)")
    print(spectra_rec[idx, :5], '...', spectra_rec[idx, -5:])

if __name__ == "__main__":
    main()