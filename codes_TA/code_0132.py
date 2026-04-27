#!/usr/bin/env python3
import numpy as np
from scipy.special import erf
from sklearn.linear_model import Ridge

# ----------------------------
# 1. Define spectral basis
# ----------------------------
def gaussian_basis(wl, centers, widths):
    """Return a matrix of Gaussian basis functions."""
    G = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths[None, :])**2)
    return G / np.sqrt(2 * np.pi) / widths[None, :]

def create_basis():
    wl = np.arange(4000, 8001, 10)          # Angstroms
    centers = np.linspace(4200, 7800, 5)    # 5 Gaussians
    widths = np.full_like(centers, 200.0)   # 200 Å width
    basis = gaussian_basis(wl, centers, widths)
    return wl, basis

# ----------------------------
# 2. Generate synthetic spectra
# ----------------------------
def generate_synthetic_spectra(n_spectra, basis, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    coeffs = rng.normal(loc=0.0, scale=1.0, size=(n_spectra, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs

# ----------------------------
# 3. Define filters
# ----------------------------
def define_filters(wl):
    """Simple top-hat filters."""
    filt_specs = {
        'U': (3600, 4300),
        'B': (4300, 5200),
        'V': (5200, 6000),
        'R': (6000, 7000)
    }
    filters = {}
    for name, (lo, hi) in filt_specs.items():
        trans = np.where((wl >= lo) & (wl <= hi), 1.0, 0.0)
        filters[name] = trans
    return filters

# ----------------------------
# 4. Compute photometry
# ----------------------------
def compute_photometry(spectra, filters, wl):
    """Integrate spectra over each filter."""
    n_spec = spectra.shape[0]
    n_filt = len(filters)
    flux = np.zeros((n_spec, n_filt))
    for i, (name, trans) in enumerate(filters.items()):
        integrand = spectra * trans
        flux[:, i] = np.trapz(integrand, wl, axis=1)
    return flux

# ----------------------------
# 5. Reconstruct spectra
# ----------------------------
def build_design_matrix(basis, filters, wl):
    """Design matrix mapping basis coeffs to filter fluxes."""
    n_filt = len(filters)
    n_basis = basis.shape[1]
    M = np.zeros((n_filt, n_basis))
    for i, trans in enumerate(filters.values()):
        for j in range(n_basis):
            integrand = basis[:, j] * trans
            M[i, j] = np.trapz(integrand, wl)
    return M

def reconstruct_coefficients(flux, M, alpha=1e-2):
    """Solve ridge regression to recover basis coefficients."""
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(M, flux.T)
    return ridge.coef_.T

def reconstruct_spectra(coeffs, basis):
    return coeffs @ basis.T

# ----------------------------
# Main routine
# ----------------------------
def main():
    rng = np.random.default_rng(seed=42)

    # Build model
    wl, basis = create_basis()

    # Generate spectra
    n_spectra = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(n_spectra, basis, rng)

    # Filters
    filters = define_filters(wl)

    # Photometry
    flux_obs = compute_photometry(spectra_true, filters, wl)

    # Reconstruction
    M = build_design_matrix(basis, filters, wl)
    coeffs_rec = reconstruct_coefficients(flux_obs, M, alpha=1e-2)
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # Simple error metric
    error = np.linalg.norm(spectra_true - spectra_rec, axis=1)
    print(f"Mean reconstruction error (norm per spectrum): {error.mean():.4f}")

if __name__ == "__main__":
    main()