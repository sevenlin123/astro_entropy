#!/usr/bin/env python3
import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import LinearRegression

def generate_wavelength_grid(n_points=200, wave_min=400, wave_max=800):
    """Create a uniform wavelength grid in nm."""
    return np.linspace(wave_min, wave_max, n_points)


def create_gaussian_basis(grid, centers, width=20):
    """
    Create Gaussian basis functions on the grid.
    
    Parameters
    ----------
    grid : ndarray
        Wavelength grid.
    centers : list or array
        Centers of the Gaussian components (nm).
    width : float
        Standard deviation of Gaussians (nm).
    
    Returns
    -------
    basis : ndarray shape (n_basis, n_grid)
        Basis functions evaluated on the grid.
    """
    basis = []
    for cen in centers:
        gauss = np.exp(-0.5 * ((grid - cen) / width)**2)
        basis.append(gauss)
    return np.array(basis)


def generate_random_spectra(basis, n_samples=50, coeff_std=1.0, random_state=None):
    """
    Generate synthetic spectra as linear combinations of basis functions.
    
    Parameters
    ----------
    basis : ndarray (n_basis, n_grid)
        Basis functions.
    n_samples : int
        Number of synthetic spectra.
    coeff_std : float
        Standard deviation for coefficient sampling.
    random_state : int or None
        Seed for reproducibility.
    
    Returns
    -------
    spectra : ndarray (n_samples, n_grid)
        Synthetic spectra.
    coeffs : ndarray (n_samples, n_basis)
        True coefficients used.
    """
    rng = np.random.default_rng(random_state)
    coeffs = rng.normal(scale=coeff_std, size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis
    return spectra, coeffs


def create_top_hat_filters(grid, filter_ranges):
    """
    Create simple top‑hat filter transmission curves.
    
    Parameters
    ----------
    grid : ndarray
        Wavelength grid.
    filter_ranges : list of tuples
        Each tuple is (wave_start, wave_end) in nm.
    
    Returns
    -------
    filters : ndarray (n_filters, n_grid)
        Filter transmissions.
    """
    filters = []
    for w_start, w_end in filter_ranges:
        trans = np.zeros_like(grid)
        mask = (grid >= w_start) & (grid <= w_end)
        trans[mask] = 1.0
        filters.append(trans)
    return np.array(filters)


def integrate_over_filters(spectra, filters, dx):
    """
    Compute synthetic photometry by integrating spectra over filters.
    
    Parameters
    ----------
    spectra : ndarray (n_samples, n_grid)
    filters : ndarray (n_filters, n_grid)
    dx : float
        Wavelength step size (nm).
    
    Returns
    -------
    photometry : ndarray (n_samples, n_filters)
    """
    # multiply spectra with filters elementwise and sum over wavelength
    phot = spectra[:, :, None] * filters[None, :, :]  # shape (n_samples, n_filters, n_grid)
    return (phot.sum(axis=2) * dx)


def build_design_matrix(basis, filters, dx):
    """
    Build design matrix that maps basis coefficients to photometric fluxes.
    
    Parameters
    ----------
    basis : ndarray (n_basis, n_grid)
    filters : ndarray (n_filters, n_grid)
    dx : float
    
    Returns
    -------
    X : ndarray (n_filters, n_basis)
    """
    # each filter response to a basis function
    X = np.array([np.trapz(basis * filt, dx=dx) for filt in filters])
    return X.T  # shape (n_basis, n_filters)


def reconstruct_coefficients(photometry, design_matrix):
    """
    Reconstruct basis coefficients from photometric data via least squares.
    
    Parameters
    ----------
    photometry : ndarray (n_samples, n_filters)
    design_matrix : ndarray (n_filters, n_basis)
    
    Returns
    -------
    coeffs_rec : ndarray (n_samples, n_basis)
    """
    # Transpose design matrix for sklearn which expects (samples, features)
    X = design_matrix.T  # shape (n_filters, n_basis) -> (n_filters, n_basis)
    reg = LinearRegression(fit_intercept=False)
    coeffs_rec = reg.fit(X, photometry.T).coef_.T  # shape (n_samples, n_basis)
    return coeffs_rec


def reconstruct_spectrum(coeffs_rec, basis):
    """
    Reconstruct spectra from recovered coefficients.
    
    Parameters
    ----------
    coeffs_rec : ndarray (n_samples, n_basis)
    basis : ndarray (n_basis, n_grid)
    
    Returns
    -------
    spectra_rec : ndarray (n_samples, n_grid)
    """
    return coeffs_rec @ basis


def main():
    # Settings
    rng_seed = 42
    n_samples = 30
    n_basis = 4
    n_filters = 3
    centers = [420, 520, 620, 720]
    filter_ranges = [(400, 500), (500, 600), (600, 700)]
    
    # Generate wavelength grid
    grid = generate_wavelength_grid()
    dx = grid[1] - grid[0]
    
    # Create basis functions
    basis = create_gaussian_basis(grid, centers, width=15)
    
    # Generate synthetic spectra and true coefficients
    spectra_true, coeffs_true = generate_random_spectra(
        basis, n_samples=n_samples, coeff_std=1.0, random_state=rng_seed
    )
    
    # Create filters
    filters = create_top_hat_filters(grid, filter_ranges)
    
    # Compute synthetic photometry
    photometry = integrate_over_filters(spectra_true, filters, dx)
    
    # Build design matrix
    X = build_design_matrix(basis, filters, dx)
    
    # Reconstruct coefficients from photometry
    coeffs_rec = reconstruct_coefficients(photometry, X)
    
    # Reconstruct spectra
    spectra_rec = reconstruct_spectrum(coeffs_rec, basis)
    
    # Print comparison for first sample
    idx = 0
    print("True coefficients:", coeffs_true[idx])
    print("Reconstructed coefficients:", coeffs_rec[idx])
    print("Difference norm:", np.linalg.norm(coeffs_true[idx] - coeffs_rec[idx]))
    print("Spectral reconstruction error (L2):",
          np.linalg.norm(spectra_true[idx] - spectra_rec[idx]) / np.linalg.norm(spectra_true[idx]))

if __name__ == "__main__":
    main()