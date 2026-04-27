#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def gaussian_basis(wavelengths, center, width):
    """Return a single Gaussian basis function."""
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def build_basis(wavelengths, n_basis=4):
    """Generate a set of Gaussian basis functions."""
    rng = np.random.default_rng(42)
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = rng.uniform(20, 50, size=n_basis)
    basis = np.array([gaussian_basis(wavelengths, c, w) for c, w in zip(centers, widths)])
    return basis  # shape (n_basis, n_wavelengths)

# ---------- Filter curves ----------
def box_filter(wavelengths, low, high):
    """Return a simple box filter curve."""
    return ((wavelengths >= low) & (wavelengths <= high)).astype(float)

def build_filters(wavelengths, n_filters=3):
    """Generate a few overlapping box filters."""
    rng = np.random.default_rng(24)
    lows = rng.uniform(wavelengths.min(), wavelengths.max() - 100, size=n_filters)
    highs = lows + rng.uniform(50, 120, size=n_filters)
    filters = np.array([box_filter(wavelengths, l, h) for l, h in zip(lows, highs)])
    return filters  # shape (n_filters, n_wavelengths)

# ---------- Synthetic data generation ----------
def generate_coefficients(n_samples, n_basis, rng=None):
    """Sample random coefficients for the basis functions."""
    rng = rng or np.random.default_rng()
    return rng.normal(scale=1.0, size=(n_samples, n_basis))

def synthesize_spectra(coeffs, basis):
    """Compute spectra as linear combinations of basis functions."""
    return coeffs @ basis  # shape (n_samples, n_wavelengths)

def compute_design_matrix(filters, basis):
    """Integrate basis functions over each filter to build design matrix."""
    # Each element B[j,i] = integral(basis_i * filter_j) ≈ sum over wavelengths
    return filters @ basis.T  # shape (n_filters, n_basis)

def synthesize_photometry(coeffs, design_matrix):
    """Compute photometric fluxes from coefficients."""
    return coeffs @ design_matrix.T  # shape (n_samples, n_filters)

# ---------- Reconstruction ----------
def reconstruct_coefficients(photometry, design_matrix, method='least_squares'):
    """
    Recover spectral coefficients from photometry.
    Method can be 'least_squares' or 'linear_regression'.
    """
    if method == 'least_squares':
        # Solve min ||A c - y||
        coeffs_rec, *_ = np.linalg.lstsq(design_matrix, photometry.T, rcond=None)
        return coeffs_rec.T  # shape (n_samples, n_basis)
    elif method == 'linear_regression':
        lr = LinearRegression(fit_intercept=False)
        lr.fit(design_matrix, photometry.T)
        return lr.coef_.T
    else:
        raise ValueError("Unknown method")

def reconstruct_spectra(coeffs_rec, basis):
    """Reconstruct spectra from recovered coefficients."""
    return coeffs_rec @ basis

# ---------- Demo ----------
def main():
    # Wavelength grid (nm)
    wavelengths = np.arange(400, 801, 10)  # 400-800 nm in 10 nm steps

    # Build basis and filters
    basis = build_basis(wavelengths, n_basis=4)          # shape (4, Nλ)
    filters = build_filters(wavelengths, n_filters=3)    # shape (3, Nλ)

    # Design matrix
    design_matrix = compute_design_matrix(filters, basis)  # shape (3, 4)

    # Generate synthetic spectra
    rng = np.random.default_rng(12345)
    n_samples = 100
    coeffs_true = generate_coefficients(n_samples, basis.shape[0], rng=rng)
    spectra_true = synthesize_spectra(coeffs_true, basis)  # shape (100, Nλ)

    # Generate photometric data
    photometry = synthesize_photometry(coeffs_true, design_matrix)  # shape (100, 3)

    # Reconstruct coefficients
    coeffs_recon = reconstruct_coefficients(photometry, design_matrix, method='least_squares')

    # Reconstruct spectra
    spectra_recon = reconstruct_spectra(coeffs_recon, basis)

    # Compare one example
    idx = 0
    print("Original spectrum (first 10 points):")
    print(spectra_true[idx, :10])
    print("\nReconstructed spectrum (first 10 points):")
    print(spectra_recon[idx, :10])

    # Simple error metric
    mse = np.mean((spectra_true - spectra_recon)**2)
    print(f"\nMean squared error over all spectra: {mse:.6f}")

if __name__ == "__main__":
    main()