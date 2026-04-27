#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.
"""

import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Spectral model (basis functions)
# ----------------------------------------------------------------------
def wavelength_grid(start=400, stop=800, steps=401):
    """Create a linear wavelength grid (nm)."""
    return np.linspace(start, stop, steps)

def gaussian_basis(center, width, wv):
    """Return a Gaussian basis function evaluated on wv."""
    return np.exp(-0.5 * ((wv - center) / width)**2)

def create_basis_functions(n_basis=5, wv=None):
    """
    Create a list of Gaussian basis functions.
    Centers are evenly spaced across the wavelength range.
    """
    if wv is None:
        wv = wavelength_grid()
    centers = np.linspace(wv[0], wv[-1], n_basis)
    width = (wv[-1] - wv[0]) / (n_basis * 2)
    return [gaussian_basis(c, width, wv) for c in centers]

# ----------------------------------------------------------------------
# 2. Synthetic spectra generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis_funcs):
    """
    Generate synthetic spectra as random linear combinations
    of the provided basis functions.
    Returns:
        spectra : (n_samples, n_wave) array
        coeffs  : (n_samples, n_basis) array
    """
    n_basis = len(basis_funcs)
    n_wave = basis_funcs[0].size
    coeffs = np.random.randn(n_samples, n_basis)
    basis_matrix = np.vstack(basis_funcs).T   # (n_wave, n_basis)
    spectra = coeffs @ basis_matrix.T         # (n_samples, n_wave)
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Photometric data generation
# ----------------------------------------------------------------------
def gaussian_filter(center, width, wv):
    """Return a Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wv - center) / width)**2)

def create_filters(filter_centers, filter_width, wv=None):
    """
    Create a list of filter transmission curves.
    filter_centers: iterable of central wavelengths
    filter_width: common width of all filters
    """
    if wv is None:
        wv = wavelength_grid()
    return [gaussian_filter(c, filter_width, wv) for c in filter_centers]

def compute_photometry(spectra, filters):
    """
    Compute synthetic photometry by integrating each spectrum over each filter.
    Normalise each filter by its total transmission.
    Returns:
        photometry : (n_samples, n_filters) array
    """
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    photometry = np.zeros((n_samples, n_filters))
    for j, filt in enumerate(filters):
        norm = trapz(filt, x=wavelength_grid())
        for i in range(n_samples):
            photometry[i, j] = trapz(spectra[i] * filt, x=wavelength_grid()) / norm
    return photometry

# ----------------------------------------------------------------------
# 4. Spectrum reconstruction from photometry
# ----------------------------------------------------------------------
def build_design_matrix(filters, basis_funcs):
    """
    Build the linear mapping from basis coefficients to photometry.
    A_{ji} = integral(filter_j * basis_i) / integral(filter_j)
    Returns:
        design_matrix : (n_filters, n_basis) array
    """
    n_filters = len(filters)
    n_basis = len(basis_funcs)
    design = np.empty((n_filters, n_basis))
    wv = wavelength_grid()
    for j, filt in enumerate(filters):
        norm = trapz(filt, x=wv)
        for i, base in enumerate(basis_funcs):
            design[j, i] = trapz(base * filt, x=wv) / norm
    return design

def reconstruct_spectra(photometry, basis_funcs, filters):
    """
    Reconstruct spectra from photometry.
    Returns:
        reconstructed_spectra : (n_samples, n_wave)
        recovered_coeffs      : (n_samples, n_basis)
    """
    design = build_design_matrix(filters, basis_funcs)  # (n_filt, n_basis)
    # Solve for coefficients using linear regression (least squares)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design.T, photometry.T)
    coeffs_recon = reg.coef_.T          # (n_samples, n_basis)
    # Reconstruct spectra
    basis_matrix = np.vstack(basis_funcs).T   # (n_wave, n_basis)
    spectra_recon = coeffs_recon @ basis_matrix.T
    return spectra_recon, coeffs_recon

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Settings
    wav = wavelength_grid()
    n_samples = 100
    n_basis = 7
    n_filters = 4

    # 1. Basis functions
    bases = create_basis_functions(n_basis=n_basis, wv=wav)

    # 2. Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, bases)

    # 3. Create filters and compute photometry
    filt_centers = [450, 550, 650, 750]  # nm
    filt_width = 40                      # nm
    filters = create_filters(filt_centers, filt_width, wv=wav)
    phot = compute_photometry(spectra_true, filters)

    # 4. Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(phot, bases, filters)

    # Evaluate reconstruction
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec)**2))
    print(f"Reconstruction RMSE: {rmse:.4f}")

    # Compare true vs recovered coefficients for first sample
    print("\nTrue coefficients (first sample):", coeffs_true[0])
    print("Recovered coefficients (first sample):", coeffs_rec[0])