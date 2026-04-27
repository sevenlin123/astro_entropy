#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Define spectral model (basis functions)
# ------------------------------------------------------------------
def gaussian_basis(wl, centers, widths):
    """Return array of shape (len(centers), len(wl)) containing Gaussians."""
    return np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths[None, :])**2)

# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def synthetic_spectrum(basis, weights, noise=0.0):
    """Weighted sum of basis functions + optional Gaussian noise."""
    spec = basis.T @ weights
    if noise > 0:
        spec += np.random.normal(0, noise, size=spec.shape)
    return spec

# ------------------------------------------------------------------
# 3. Photometric data from synthetic spectra
# ------------------------------------------------------------------
def filter_top_hat(center, width, wl):
    """Simple top-hat filter transmission."""
    return np.where(np.abs(wl - center) <= width/2, 1.0, 0.0)

def photometric_flux(spectrum, filter_trans, wl):
    """Integrate spectrum weighted by filter transmission."""
    return np.trapz(spectrum * filter_trans, wl)

# ------------------------------------------------------------------
# 4. Reconstruction of spectrum from photometry
# ------------------------------------------------------------------
def reconstruct_spectrum(
    photometric_values,
    basis,
    filter_transmissions,
    regularization=1e-3
):
    """
    Solve for weights that best reproduce given photometric fluxes.
    basis: array (n_pixels, n_bases)
    filter_transmissions: list of arrays (n_filters, n_pixels)
    """
    # Build design matrix: each photometric value is integral of spectrum*filter
    # => sum_i w_j * ∫ b_j(x)*f_k(x) dx
    n_bases = basis.shape[1]
    n_filters = len(filter_transmissions)
    A = np.empty((n_filters, n_bases))
    for k, filt in enumerate(filter_transmissions):
        integrand = basis * filt  # elementwise product: (n_pixels, n_bases)
        A[k, :] = np.trapz(integrand, axis=0)  # integrate over wavelength

    # Solve linear system with ridge regularization
    clf = Ridge(alpha=regularization, fit_intercept=False, solver="svd")
    clf.fit(A, photometric_values)
    recon_weights = clf.coef_
    recon_spectrum = basis.T @ recon_weights
    return recon_spectrum, recon_weights

# ------------------------------------------------------------------
# Main simulation
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(400, 800, 1000)  # nm

    # Basis: 5 Gaussian components
    n_bases = 5
    centers = np.linspace(450, 750, n_bases)
    widths = np.full(n_bases, 30.0)
    basis = gaussian_basis(wl, centers, widths)

    # True weights
    true_weights = np.array([1.5, -0.7, 2.0, 0.3, -1.0])
    true_spectrum = synthetic_spectrum(basis, true_weights, noise=0.05)

    # Define two photometric filters
    filter_centers = [500, 650]
    filter_widths = [50, 60]
    filters = [filter_top_hat(c, w, wl) for c, w in zip(filter_centers, filter_widths)]

    # Compute photometric fluxes from true spectrum
    phot_vals = np.array([photometric_flux(true_spectrum, f, wl) for f in filters])

    # Reconstruction
    recon_spec, recon_w = reconstruct_spectrum(phot_vals, basis, filters, regularization=1e-2)

    # Plot comparison
    plt.figure(figsize=(8, 4))
    plt.plot(wl, true_spectrum, label="True Spectrum", lw=2)
    plt.plot(wl, recon_spec, '--', label="Reconstructed Spectrum", lw=2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (arbitrary units)")
    plt.legend()
    plt.title("Spectrum Reconstruction from Photometry")
    plt.tight_layout()
    plt.show()

    # Print comparison of weights
    print("True weights :", true_weights)
    print("Recovered   :", recon_w)