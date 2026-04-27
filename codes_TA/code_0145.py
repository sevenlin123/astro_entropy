#!/usr/bin/env python3
"""
Minimal spectral reconstruction example:
    * Defines a simple spectral basis model
    * Generates synthetic spectra as linear combinations of basis spectra
    * Simulates broadband photometry by integrating spectra through Gaussian filters
    * Reconstructs the spectrum from photometry using linear least‑squares
"""

import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Spectral model utilities
# ----------------------------------------------------------------------
def create_wavelength_grid(start=400.0, stop=1000.0, num=200):
    """Return an evenly spaced wavelength grid in nm."""
    return np.linspace(start, stop, num)

def gaussian_bump(wl, center, sigma, amplitude):
    """Single Gaussian spectral feature."""
    return amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2)

def generate_basis_spectra(num_basis, wl):
    """
    Generate a set of basis spectra.
    Each basis spectrum consists of a few Gaussian bumps with random parameters.
    """
    rng = np.random.default_rng(seed=42)
    basis = []
    for _ in range(num_basis):
        spec = np.zeros_like(wl)
        num_bumps = rng.integers(1, 4)
        for _ in range(num_bumps):
            center = rng.uniform(wl.min(), wl.max())
            sigma  = rng.uniform(10.0, 40.0)
            amp    = rng.uniform(0.5, 1.5)
            spec += gaussian_bump(wl, center, sigma, amp)
        basis.append(spec)
    return np.vstack(basis)  # shape (num_basis, len(wl))

def generate_random_coefficients(num_basis, rng=None):
    """Random coefficients for a synthetic spectrum."""
    if rng is None:
        rng = np.random.default_rng(seed=24)
    return rng.uniform(0.0, 1.0, size=num_basis)

def synthesize_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return basis.T @ coeffs  # shape (len(wl),)

# ----------------------------------------------------------------------
# 2. Filter and photometry utilities
# ----------------------------------------------------------------------
def gaussian_filter(wl, center, sigma):
    """Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wl - center) / sigma) ** 2)

def generate_filters(num_filters, wl):
    """Generate a set of Gaussian photometric filters."""
    rng = np.random.default_rng(seed=99)
    filters = []
    for _ in range(num_filters):
        center = rng.uniform(wl.min() + 50, wl.max() - 50)
        sigma  = rng.uniform(30.0, 80.0)
        filt   = gaussian_filter(wl, center, sigma)
        filters.append(filt)
    return np.vstack(filters)  # shape (num_filters, len(wl))

def compute_photometry(spectrum, filters, wl):
    """
    Compute broadband photometry:
        F_i = ∫ S(λ) T_i(λ) dλ / ∫ T_i(λ) dλ
    """
    numerators   = np.array([simps(spectrum * filt, wl) for filt in filters])
    denominators = np.array([simps(filt, wl) for filt in filters])
    return numerators / denominators

# ----------------------------------------------------------------------
# 3. Reconstruction utilities
# ----------------------------------------------------------------------
def build_design_matrix(filters, basis, wl):
    """
    Build matrix A where A_ij = ∫ B_j(λ) T_i(λ) dλ / ∫ T_i(λ) dλ
    """
    num_filters = filters.shape[0]
    num_basis   = basis.shape[0]
    A = np.empty((num_filters, num_basis))
    denom = np.array([simps(filt, wl) for filt in filters])  # shape (num_filters,)
    for i, filt in enumerate(filters):
        for j, base in enumerate(basis):
            A[i, j] = simps(base * filt, wl) / denom[i]
    return A

def reconstruct_spectrum_from_photometry(photon, filters, basis, wl):
    """
    Estimate spectrum coefficients via linear least squares and reconstruct spectrum.
    """
    A = build_design_matrix(filters, basis, wl)
    reg = LinearRegression(fit_intercept=False).fit(A, photon)
    coeffs = reg.coef_
    recon_spec = synthesize_spectrum(basis, coeffs)
    return recon_spec, coeffs

# ----------------------------------------------------------------------
# 4. Example workflow
# ----------------------------------------------------------------------
def main():
    # Parameters
    NUM_BASIS = 5
    NUM_FILTERS = 4

    # 1. Wavelength grid
    wl = create_wavelength_grid()

    # 2. Basis spectra
    basis = generate_basis_spectra(NUM_BASIS, wl)

    # 3. Synthetic spectrum
    rng = np.random.default_rng(seed=101)
    true_coeffs = generate_random_coefficients(NUM_BASIS, rng)
    true_spectrum = synthesize_spectrum(basis, true_coeffs)

    # 4. Filters
    filters = generate_filters(NUM_FILTERS, wl)

    # 5. Generate photometry
    photometry = compute_photometry(true_spectrum, filters, wl)

    # 6. Reconstruct spectrum
    recon_spectrum, recon_coeffs = reconstruct_spectrum_from_photometry(
        photometry, filters, basis, wl
    )

    # 7. Simple comparison output
    print("True coefficients :", true_coeffs)
    print("Recovered coeffs  :", recon_coeffs)
    error = np.linalg.norm(recon_spectrum - true_spectrum) / np.linalg.norm(true_spectrum)
    print(f"Relative L2 error between true and reconstructed spectrum: {error:.3e}")

if __name__ == "__main__":
    main()