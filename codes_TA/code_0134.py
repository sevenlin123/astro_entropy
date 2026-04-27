#!/usr/bin/env python3
import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model – Gaussian basis functions
# ----------------------------------------------------------------------
def gaussian_basis(wl, centers, sigma):
    """Return an array of Gaussian basis functions."""
    basis = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / sigma) ** 2)
    return basis  # shape (N_wl, N_basis)

# ----------------------------------------------------------------------
# Filter responses – Gaussian bandpasses
# ----------------------------------------------------------------------
def filter_response(wl, centers, widths):
    """Return an array of Gaussian filter responses."""
    filt = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths) ** 2)
    return filt  # shape (N_wl, N_filters)

# ----------------------------------------------------------------------
# Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wl, basis, noise_level=0.01):
    """Generate synthetic spectra as linear combos of basis functions."""
    n_basis = basis.shape[1]
    coeffs = np.random.randn(n_samples, n_basis)  # random coefficients
    spectra = coeffs @ basis.T                     # shape (n_samples, N_wl)
    spectra += noise_level * np.std(spectra) * np.random.randn(*spectra.shape)
    return spectra, coeffs

# ----------------------------------------------------------------------
# Generate photometric data from spectra
# ----------------------------------------------------------------------
def generate_photometry(spectra, filt, wl):
    """Integrate spectra through filter responses."""
    n_samples = spectra.shape[0]
    n_filters = filt.shape[1]
    phot = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        # Normalized filter integral
        norm = trapz(filt[:, i], wl)
        phot[:, i] = trapz(spectra * filt[:, i][None, :], wl, axis=1) / norm
    return phot

# ----------------------------------------------------------------------
# Reconstruct spectra from photometry
# ----------------------------------------------------------------------
def reconstruct_from_photometry(phot, basis, filt, wl):
    """
    Compute coefficients that best reproduce the photometry,
    then reconstruct the spectra.
    """
    # Build the forward matrix M: M_{ij} = <basis_j * filt_i> / <filt_i>
    n_filters = filt.shape[1]
    n_basis   = basis.shape[1]
    M = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        denom = trapz(filt[:, i], wl)
        for j in range(n_basis):
            M[i, j] = trapz(basis[:, j] * filt[:, i], wl) / denom

    # Use linear regression to map photometry to coefficients
    # (regressor = M^-1 but computed via least‑squares)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(phot, np.eye(n_basis))   # fit mapping phot -> identity matrix
    # The learned weights give the inverse transform
    inv_M = reg.coef_.T
    coeffs_est = inv_M @ phot.T       # shape (n_basis, n_samples)
    coeffs_est = coeffs_est.T         # shape (n_samples, n_basis)

    # Reconstruct spectra
    spectra_rec = coeffs_est @ basis.T
    return spectra_rec, coeffs_est

# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (400–800 nm)
    wl = np.linspace(400, 800, 1000)

    # Define basis functions
    basis_centers = np.linspace(450, 650, 5)  # 5 Gaussian peaks
    basis_sigma   = 20.0
    basis = gaussian_basis(wl, basis_centers, basis_sigma)

    # Define filter responses
    filt_centers = np.array([450, 520, 590])
    filt_widths  = np.array([40.0, 40.0, 40.0])
    filt = filter_response(wl, filt_centers, filt_widths)

    # Generate synthetic data
    n_samples = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_samples, wl, basis, noise_level=0.02
    )
    phot = generate_photometry(spectra_true, filt, wl)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_from_photometry(phot, basis, filt, wl)

    # Simple error assessment
    err = np.mean((spectra_true - spectra_rec)**2)
    print(f"Mean squared reconstruction error: {err:.4e}")

    # Plot one example (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(8, 4))
        plt.plot(wl, spectra_true[idx], label="True Spectrum")
        plt.plot(wl, spectra_rec[idx], '--', label="Reconstructed Spectrum")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux")
        plt.title("Spectral Reconstruction Example")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass