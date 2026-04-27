#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def wavelength_grid(start=300.0, end=1000.0, step=1.0):
    """Generate a wavelength array (nm)."""
    return np.arange(start, end + step, step)


def generate_filters(grid, n_filters=5, seed=42):
    """Create a list of random Gaussian filter transmission curves."""
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(n_filters):
        mu = rng.uniform(grid[0], grid[-1])
        sigma = rng.uniform(15.0, 35.0)
        trans = np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
        trans /= trans.max()  # normalise to peak 1
        filters.append(trans)
    return filters


def generate_synthetic_spectrum(grid, n_components=5, seed=7):
    """Create a synthetic spectrum as a sum of random Gaussians."""
    rng = np.random.default_rng(seed)
    spectrum = np.zeros_like(grid)
    for _ in range(n_components):
        mu = rng.uniform(grid[0], grid[-1])
        sigma = rng.uniform(10.0, 30.0)
        amp = rng.uniform(0.5, 1.5)
        spectrum += amp * np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
    # Optional: add a small background
    spectrum += rng.uniform(0.05, 0.1)
    return spectrum


def photometric_flux(spectrum, filters):
    """Integrate a spectrum over each filter transmission curve."""
    # Assume unit wavelength spacing; flux ∝ sum(s * T)
    return np.array([np.sum(spectrum * filt) for filt in filters])


def reconstruct_spectrum(ph_flux, filters, grid, alpha=0.0):
    """
    Reconstruct the spectrum from photometric fluxes using linear regression.
    Returns the estimated spectrum on the input wavelength grid.
    """
    R = np.array(filters)                # shape (M, N)
    # Add a small L2 regularisation term if desired
    if alpha > 0.0:
        # Solve (R^T R + alpha I) beta = R^T y
        A = R.T @ R + alpha * np.eye(R.shape[1])
        b = R.T @ ph_flux
        beta = np.linalg.solve(A, b)
    else:
        # Ordinary least squares via sklearn
        lr = LinearRegression(fit_intercept=False)
        lr.fit(R, ph_flux)
        beta = lr.coef_
    return beta


def main():
    # 1. Define the wavelength grid
    grid = wavelength_grid()

    # 2. Generate a synthetic spectrum
    true_spec = generate_synthetic_spectrum(grid)

    # 3. Create filter responses
    filters = generate_filters(grid, n_filters=6)

    # 4. Generate photometric data
    ph_flux = photometric_flux(true_spec, filters)

    # 5. Reconstruct the spectrum
    recon_spec = reconstruct_spectrum(ph_flux, filters, grid, alpha=1e-4)

    # 6. Evaluate reconstruction quality
    rel_err = np.linalg.norm(recon_spec - true_spec) / np.linalg.norm(true_spec)
    print(f"Relative reconstruction error: {rel_err:.4f}")

    # 7. Compare a few points
    indices = np.linspace(0, len(grid) - 1, 5, dtype=int)
    print("\nWavelength (nm) | True | Reconstructed")
    for i in indices:
        print(f"{grid[i]:>12.1f}   | {true_spec[i]:.4f} | {recon_spec[i]:.4f}")


if __name__ == "__main__":
    main()