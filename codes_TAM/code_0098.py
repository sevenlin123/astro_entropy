#!/usr/bin/env python3
"""
Minimal spectral reconstruction demo:
- Define a simple spectral basis (Gaussian functions)
- Generate synthetic spectra as random linear combinations of the basis
- Define a few top-hat photometric filters
- Compute synthetic photometry by integrating the spectrum over each filter
- Reconstruct the spectrum from photometry by solving a linear least‑squares problem
"""

import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# ---------- Spectral Model ----------
def create_wavelength_grid(start=3000, stop=10000, n_points=500):
    """Create a logarithmically spaced wavelength grid (Angstrom)."""
    return np.logspace(np.log10(start), np.log10(stop), n_points)

def gaussian_basis(center, width, wls):
    """Return a normalized Gaussian function evaluated on wls."""
    return np.exp(-0.5 * ((wls - center) / width)**2)

def create_basis(wls, centers, widths):
    """Build a matrix of basis functions evaluated on wls."""
    return np.array([gaussian_basis(c, w, wls) for c, w in zip(centers, widths)]).T

def generate_synthetic_spectrum(basis, coef_std=1.0, random_state=None):
    """Generate a synthetic spectrum as a random linear combination of basis functions."""
    rng = np.random.default_rng(random_state)
    coeffs = rng.normal(scale=coef_std, size=basis.shape[1])
    return basis @ coeffs, coeffs

# ---------- Photometric Filters ----------
def top_hat_filter(wls, center, width):
    """Return a top‑hat filter transmission curve."""
    return np.where((wls >= center - width/2) & (wls <= center + width/2), 1.0, 0.0)

def create_filters(wls, centers, widths):
    """Create a list of filter transmission curves."""
    return [top_hat_filter(wls, c, w) for c, w in zip(centers, widths)]

def compute_photometry(spectrum, filters, wls):
    """Compute synthetic photometry by integrating the spectrum over each filter."""
    return np.array([np.trapz(spectrum * filt, wls) for filt in filters])

# ---------- Spectrum Reconstruction ----------
def reconstruct_spectrum(photometry, basis, filters, wls):
    """
    Reconstruct the spectrum from photometry by solving:
        photometry ≈ (∫ basis_i * filter_j dλ ) * coeff_i
    """
    # Build the design matrix A where A[j,i] = ∫ basis_i * filter_j dλ
    A = np.array([
        [np.trapz(basis[:, i] * filt, wls) for i in range(basis.shape[1])]
        for filt in filters
    ])
    # Fit linear regression without intercept
    lr = LinearRegression(fit_intercept=False).fit(A, photometry)
    coeffs = lr.coef_
    return basis @ coeffs, coeffs

# ---------- Demo ----------
def main():
    # Wavelength grid
    wls = create_wavelength_grid()

    # Basis functions
    basis_centers = np.linspace(4000, 9000, 8)          # 8 Gaussian components
    basis_widths   = np.full_like(basis_centers, 600)   # Fixed width
    basis = create_basis(wls, basis_centers, basis_widths)

    # Synthetic spectrum
    true_spectrum, true_coeffs = generate_synthetic_spectrum(basis, random_state=42)

    # Photometric filters
    filter_centers = np.array([3500, 4500, 5500, 6500, 7500])   # 5 filters
    filter_widths  = np.full_like(filter_centers, 800)          # Fixed width
    filters = create_filters(wls, filter_centers, filter_widths)

    # Synthetic photometry
    photometry = compute_photometry(true_spectrum, filters, wls)

    # Reconstruct spectrum
    recon_spectrum, recon_coeffs = reconstruct_spectrum(photometry, basis, filters, wls)

    # Simple diagnostics
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(wls, true_spectrum, label='True spectrum')
    plt.plot(wls, recon_spectrum, '--', label='Reconstructed spectrum')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Spectral Reconstruction Demo')
    plt.legend()
    plt.show()

    print("True coefficients:\n", true_coeffs)
    print("\nReconstructed coefficients:\n", recon_coeffs)

if __name__ == "__main__":
    main()