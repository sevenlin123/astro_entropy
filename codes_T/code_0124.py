#!/usr/bin/env python3
import numpy as np
from scipy.constants import c, h, k
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# Spectral model: simple Planck black‑body spectrum
def planck_lambda(wl_m, temperature):
    """Planck function B(λ,T) in W·m⁻²·sr⁻¹·µm⁻¹."""
    wl_m = np.asarray(wl_m)
    numerator = 2.0 * h * c**2 / wl_m**5
    denominator = np.exp(h * c / (wl_m * k * temperature)) - 1.0
    return numerator / denominator

# ------------------------------------------------------------------
# Generate synthetic spectra by perturbing a base black‑body
def generate_synthetic_spectra(n_spec, wl_grid, base_temp=6000.0):
    """Create n_spec synthetic spectra by adding random Gaussian bumps."""
    n_wl = len(wl_grid)
    spectra = []
    base = planck_lambda(wl_grid, base_temp)

    rng = np.random.default_rng()
    for _ in range(n_spec):
        amp = rng.normal(scale=0.1 * base.max())
        center = rng.uniform(wl_grid.min(), wl_grid.max())
        sigma = rng.uniform((wl_grid.max()-wl_grid.min())/20,
                            (wl_grid.max()-wl_grid.min())/5)
        bump = amp * np.exp(-0.5 * ((wl_grid-center)/sigma)**2)
        spectra.append(base + bump)
    return np.vstack(spectra)  # shape (n_spec, n_wl)

# ------------------------------------------------------------------
# Generate simple Gaussian filter curves
def generate_gaussian_filters(n_filter, wl_grid, width_factor=0.05):
    """Generate n_filter Gaussian bandpasses."""
    filters = []
    rng = np.random.default_rng()
    for _ in range(n_filter):
        center = rng.uniform(wl_grid.min(), wl_grid.max())
        sigma = width_factor * (wl_grid.max() - wl_grid.min())
        filt = np.exp(-0.5 * ((wl_grid-center)/sigma)**2)
        filt /= filt.sum()          # normalize area to 1
        filters.append(filt)
    return np.vstack(filters)      # shape (n_filter, n_wl)

# ------------------------------------------------------------------
# Compute photometric fluxes by integrating spectra through filters
def compute_photometry(spectra, filters, wl_grid):
    """Return photometry array (n_filter,) for a single spectrum."""
    fluxes = []
    for filt in filters:
        flux = np.trapz(spectra * filt, wl_grid)
        fluxes.append(flux)
    return np.array(fluxes)

# ------------------------------------------------------------------
# Build basis matrix for photometry
def build_basis_matrix(basis_spectra, filters, wl_grid):
    """Return matrix A (n_filter, n_basis) where A[j,i]=∫basis_i*filt_j."""
    n_filter = filters.shape[0]
    n_basis  = basis_spectra.shape[0]
    A = np.zeros((n_filter, n_basis))
    for i in range(n_basis):
        for j in range(n_filter):
            A[j, i] = np.trapz(basis_spectra[i] * filters[j], wl_grid)
    return A

# ------------------------------------------------------------------
# Reconstruct a spectrum from photometric data
def reconstruct_spectrum(photon_flux, filters, basis_spectra, wl_grid,
                         alpha=1e-2):
    """Reconstruct spectrum using Ridge regression on photometric basis."""
    A = build_basis_matrix(basis_spectra, filters, wl_grid)
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(A, photon_flux)
    coeffs = ridge.coef_          # shape (n_basis,)
    recon_spec = np.dot(coeffs, basis_spectra)  # shape (n_wl,)
    return recon_spec, coeffs

# ------------------------------------------------------------------
# Main routine – synthetic test case
def main():
    # Wavelength grid in meters (visible range)
    wl_grid = np.linspace(400e-9, 700e-9, 300)   # 400–700 nm

    # Generate synthetic training spectra (used as basis)
    n_basis = 10
    basis_spectra = generate_synthetic_spectra(n_basis, wl_grid,
                                               base_temp=5500.0)

    # Generate Gaussian filters
    n_filters = 5
    filters = generate_gaussian_filters(n_filters, wl_grid)

    # Generate one target spectrum (different temperature)
    target_temp = 5000.0
    target_spec = planck_lambda(wl_grid, target_temp)

    # Compute photometric data for the target
    photometry_target = compute_photometry(target_spec, filters, wl_grid)

    # Reconstruct spectrum from photometry
    recon_spec, coeffs = reconstruct_spectrum(photometry_target,
                                              filters, basis_spectra,
                                              wl_grid, alpha=1e-3)

    # Compare: compute photometry of reconstructed spectrum
    photometry_recon = compute_photometry(recon_spec, filters, wl_grid)

    # Print results
    print("Target photometry:", photometry_target)
    print("Reconstructed photometry:", photometry_recon)
    print("Flux residual norm:", np.linalg.norm(photometry_target - photometry_recon))

    # Optional: compare spectra
    diff_norm = np.linalg.norm(target_spec - recon_spec)
    print("Spectrum difference L2 norm:", diff_norm)

if __name__ == "__main__":
    main()