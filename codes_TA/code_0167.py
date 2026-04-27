#!/usr/bin/env python3
import numpy as np
from numpy import trapz
from sklearn.linear_model import Ridge

def wavelength_grid(n_points=2000, w_min=400.0, w_max=800.0):
    """Create a uniform wavelength grid in nanometers."""
    return np.linspace(w_min, w_max, n_points)

def gaussian_basis(wl, centers, widths):
    """
    Evaluate a set of Gaussian basis functions on a wavelength grid.
    Returns an array of shape (n_basis, len(wl)).
    """
    gauss = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths[None, :])**2)
    return gauss

def generate_synthetic_spectrum(wl, centers, widths, coeffs):
    """
    Construct a synthetic spectrum as a linear combination of Gaussian bases.
    """
    basis = gaussian_basis(wl, centers, widths)
    return basis.T @ coeffs

def filter_response_top_hat(wl, center, width):
    """
    Create a top‑hat filter centered at 'center' with full width 'width'.
    """
    mask = (wl >= center - width/2) & (wl <= center + width/2)
    resp = np.zeros_like(wl)
    resp[mask] = 1.0
    return resp

def build_filters(wl):
    """
    Define a set of synthetic photometric filters.
    Returns a list of filter response arrays.
    """
    centers = [450.0, 550.0, 650.0]
    width   = 20.0  # nm
    return [filter_response_top_hat(wl, c, width) for c in centers]

def photometry_from_spectrum(spectrum, filters, wl):
    """
    Integrate the spectrum through each filter to obtain photometric fluxes.
    """
    return np.array([trapz(spectrum * f, wl) for f in filters])

def reconstruct_coefficients(filters, wl, photometry, centers, widths, alpha=0.0):
    """
    Reconstruct the coefficients of the Gaussian basis from photometry.
    Uses a ridge regression solver (alpha=0 gives ordinary least squares).
    """
    # Build the projection matrix A
    basis = gaussian_basis(wl, centers, widths)  # shape (n_basis, n_wl)
    A = np.vstack([trapz(basis * f[:, None], wl) for f in filters]).T  # (n_wl, n_basis)
    # Solve for coefficients
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    ridge.fit(A, photometry)
    return ridge.coef_

def main():
    # 1. Define wavelength grid and basis
    wl = wavelength_grid()
    nbasis = 5
    np.random.seed(42)
    centers = np.linspace(420.0, 780.0, nbasis)
    widths  = np.full(nbasis, 30.0)

    # 2. Generate synthetic spectrum
    true_coeffs = np.random.randn(nbasis)
    spectrum = generate_synthetic_spectrum(wl, centers, widths, true_coeffs)

    # 3. Create filters and obtain photometry
    filters = build_filters(wl)
    phot = photometry_from_spectrum(spectrum, filters, wl)

    # Add optional Gaussian noise to photometry
    noise_level = 0.02  # relative noise
    phot_noisy = phot + np.random.normal(scale=noise_level*np.max(phot), size=phot.shape)

    # 4. Reconstruct coefficients from noisy photometry
    recon_coeffs = reconstruct_coefficients(filters, wl, phot_noisy, centers, widths, alpha=1e-3)

    # Reconstruct spectrum from recovered coefficients
    recon_spectrum = generate_synthetic_spectrum(wl, centers, widths, recon_coeffs)

    # 5. Print comparison
    print("True coefficients:\n", true_coeffs)
    print("\nRecovered coefficients:\n", recon_coeffs)
    print("\nPhotometry (true):\n", phot)
    print("\nPhotometry (noisy):\n", phot_noisy)
    print("\nPhotometry (recon):\n", photometry_from_spectrum(recon_spectrum, filters, wl))

if __name__ == "__main__":
    main()