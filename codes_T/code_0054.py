#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework:
- Generate synthetic spectra from a basis of Gaussian features.
- Compute synthetic photometry through rectangular filters.
- Reconstruct spectra from photometric fluxes via linear least‑squares.
"""

import numpy as np
from sklearn.linear_model import Ridge

# ---------------------------------------------
# Utility functions
# ---------------------------------------------

def create_wavelength_grid(start=4000, end=8000, n=200):
    """Return a 1D wavelength array (in Å)."""
    return np.linspace(start, end, n)

def gaussian(x, amp, cen, wid):
    """One‑dimensional Gaussian."""
    return amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)

def generate_basis_functions(wavelengths, n_basis, rng=None):
    """Return a list of Gaussian basis functions."""
    rng = np.random.default_rng(rng)
    bases = []
    for _ in range(n_basis):
        amp = rng.uniform(0.5, 1.5)
        cen = rng.uniform(wavelengths[0], wavelengths[-1])
        wid = rng.uniform(50, 150)
        bases.append(gaussian(wavelengths, amp, cen, wid))
    return np.array(bases)  # shape (n_basis, n_wavelengths)

def define_rectangular_filters(n_filters, wavelengths, rng=None):
    """Return a list of rectangular filter transmission curves."""
    rng = np.random.default_rng(rng)
    filters = []
    for _ in range(n_filters):
        cen = rng.uniform(wavelengths[0], wavelengths[-1])
        wid = rng.uniform(200, 400)
        filt = np.where(
            (wavelengths >= cen - wid / 2) & (wavelengths <= cen + wid / 2),
            1.0,
            0.0,
        )
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, n_wavelengths)

def spectral_model(coeffs, basis_functions):
    """Linear combination of basis functions."""
    return np.dot(coeffs, basis_functions)

def generate_synthetic_spectra(n_samples, basis_functions, rng=None):
    """Generate spectra and the corresponding true coefficients."""
    rng = np.random.default_rng(rng)
    n_basis = basis_functions.shape[0]
    coeffs = rng.normal(scale=1.0, size=(n_samples, n_basis))
    spectra = np.dot(coeffs, basis_functions.T)  # shape (n_samples, n_wavelengths)
    return spectra, coeffs

def compute_photometric_fluxes(spectra, filters, wavelengths):
    """
    Compute photometric fluxes for each spectrum and filter.
    Flux = ∫ S(λ) T(λ) dλ
    """
    # Use trapezoidal integration along wavelength axis
    fluxes = np.trapz(spectra[:, :, None] * filters[None, :, :], wavelengths, axis=-1)
    # Resulting shape: (n_samples, n_filters)
    return fluxes

def build_transfer_matrix(filters, basis_functions, wavelengths):
    """
    Build M matrix where M_ij = ∫ B_j(λ) T_i(λ) dλ
    """
    # Compute integrals analytically via trapz
    M = np.trapz(basis_functions[:, :, None] * filters[None, :, :], wavelengths, axis=-1)
    # Shape: (n_filters, n_basis)
    return M

def reconstruct_spectrum_from_photometry(flux, M, basis_functions):
    """
    Solve for coefficients via ridge regression and return reconstructed spectrum.
    """
    # Ridge regression to handle ill‑conditioning
    ridge = Ridge(alpha=1e-3, fit_intercept=False, solver='auto')
    ridge.fit(M, flux)
    coeffs_rec = ridge.coef_
    spectrum_rec = spectral_model(coeffs_rec, basis_functions)
    return spectrum_rec, coeffs_rec

# ---------------------------------------------
# Main demonstration
# ---------------------------------------------
def main():
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)

    # 1. Define wavelength grid
    wl = create_wavelength_grid()

    # 2. Generate basis functions
    n_basis = 6
    basis_funcs = generate_basis_functions(wl, n_basis, rng=rng)

    # 3. Define filters
    n_filters = 4
    filters = define_rectangular_filters(n_filters, wl, rng=rng)

    # 4. Build transfer matrix M
    M = build_transfer_matrix(filters, basis_funcs, wl)

    # 5. Generate synthetic spectra
    n_samples = 10
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis_funcs, rng=rng)

    # 6. Compute photometric fluxes
    fluxes = compute_photometric_fluxes(spectra, filters, wl)

    # 7. Reconstruct the first spectrum
    sample_idx = 0
    flux_sample = fluxes[sample_idx]
    spectrum_rec, coeffs_rec = reconstruct_spectrum_from_photometry(
        flux_sample, M, basis_funcs
    )

    # 8. Display results
    print("True coefficients for sample", sample_idx)
    print(true_coeffs[sample_idx])
    print("\nRecovered coefficients:")
    print(coeffs_rec)
    print("\nDifference norm:", np.linalg.norm(true_coeffs[sample_idx] - coeffs_rec))

if __name__ == "__main__":
    main()