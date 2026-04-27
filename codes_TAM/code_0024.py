#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# Spectral model: wavelength grid and basis functions
# ------------------------------------------------------------------
def create_wavelength_grid(start_nm, end_nm, n_points):
    """Uniform wavelength grid in nanometers."""
    return np.linspace(start_nm, end_nm, n_points)

def gaussian_basis(wl, centers, sigma=20.0):
    """
    Return an array of Gaussian basis functions.
    Each basis is evaluated on the wavelength grid 'wl'.
    """
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wl - c) / sigma)**2)
        # Normalize to unit area
        g /= np.trapz(g, wl)
        basis.append(g)
    return np.array(basis)  # shape (n_basis, n_wavelength)

# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------
def top_hat_filter(wl, center, width):
    """Simple top‑hat transmission function."""
    return ((wl >= center - width / 2) & (wl <= center + width / 2)).astype(float)

def build_filters(wl):
    """Define a few broad band filters."""
    filters = {}
    filters['B'] = top_hat_filter(wl, center=445, width=90)
    filters['V'] = top_hat_filter(wl, center=551, width=90)
    filters['R'] = top_hat_filter(wl, center=658, width=90)
    return filters

# ------------------------------------------------------------------
# Synthetic spectra generation
# ------------------------------------------------------------------
def generate_synthetic_spectra(basis, n_samples, coeff_range=(0.0, 1.0)):
    """
    Randomly generate spectra as linear combinations of basis functions.
    Returns:
        spectra: shape (n_samples, n_wavelength)
        coeffs : shape (n_samples, n_basis)
    """
    n_basis = basis.shape[0]
    coeffs = np.random.uniform(low=coeff_range[0], high=coeff_range[1],
                               size=(n_samples, n_basis))
    spectra = coeffs @ basis  # matrix multiplication
    return spectra, coeffs

# ------------------------------------------------------------------
# Photometry computation
# ------------------------------------------------------------------
def compute_photometry(spectra, filters, wl):
    """
    Integrate each spectrum through each filter.
    Returns array of shape (n_samples, n_filters).
    """
    n_samples = spectra.shape[0]
    filter_names = list(filters.keys())
    phot = np.empty((n_samples, len(filter_names)))
    for i, f in enumerate(filter_names):
        trans = filters[f]
        # Integral of spectrum * transmission over wavelength
        phot[:, i] = np.trapz(spectra * trans, wl, axis=1)
    return phot, filter_names

# ------------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------------
def build_design_matrix(basis, filters, wl):
    """
    Pre‑compute A_ik = ∫ B_k(λ) * T_i(λ) dλ.
    Returns matrix of shape (n_filters, n_basis).
    """
    filter_names = list(filters.keys())
    A = np.empty((len(filter_names), basis.shape[0]))
    for i, f in enumerate(filter_names):
        trans = filters[f]
        for k in range(basis.shape[0]):
            A[i, k] = np.trapz(basis[k] * trans, wl)
    return A

def reconstruct_coefficients(phot, basis, filters, wl, alpha=1.0):
    """
    Solve A * c = phot.T for coefficients c (regularized).
    Returns reconstructed coefficients array (n_samples, n_basis).
    """
    A = build_design_matrix(basis, filters, wl)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, phot.T)          # fit on transpose for samples as columns
    coeffs_recon = ridge.coef_.T  # shape (n_samples, n_basis)
    return coeffs_recon

def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from coefficients and basis."""
    return coeffs @ basis

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
def main():
    # 1. Define wavelength grid
    wl = create_wavelength_grid(400, 800, 1000)      # 400–800 nm

    # 2. Build basis functions (5 Gaussians)
    centers = [450, 520, 590, 660, 730]
    basis = gaussian_basis(wl, centers, sigma=20.0)   # shape (5, 1000)

    # 3. Build filter set
    filters = build_filters(wl)

    # 4. Generate synthetic spectra
    n_samples = 10
    spectra_true, coeffs_true = generate_synthetic_spectra(basis, n_samples)

    # 5. Compute photometric measurements
    photometry, filter_names = compute_photometry(spectra_true, filters, wl)

    # 6. Reconstruct spectra from photometry
    coeffs_rec = reconstruct_coefficients(photometry, basis, filters, wl, alpha=0.1)
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # 7. Evaluate reconstruction error
    mse = np.mean((spectra_true - spectra_rec)**2)
    print(f"Mean squared reconstruction error: {mse:.6e}")

    # Optional: print first spectrum comparison
    idx = 0
    print("\nTrue spectrum (sample 0):")
    print(spectra_true[idx, :5], "...", spectra_true[idx, -5:])
    print("\nReconstructed spectrum (sample 0):")
    print(spectra_rec[idx, :5], "...", spectra_rec[idx, -5:])

if __name__ == "__main__":
    main()