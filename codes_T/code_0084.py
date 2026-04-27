#!/usr/bin/env python3
import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# Spectral model – 1‑D basis functions (Gaussian bumps)
# ------------------------------------------------------------------
def gaussian_basis(wl, centers, widths):
    """Return an array of shape (len(wl), len(centers))"""
    return np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths[None, :])**2)

def build_basis(wl, n_basis=5, seed=None):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(wl.min(), wl.max(), size=n_basis)
    widths  = rng.uniform((wl.max()-wl.min())/10, (wl.max()-wl.min())/5, size=n_basis)
    return gaussian_basis(wl, centers, widths)

# ------------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------------
def generate_spectra(n_samples, wl, basis, seed=None):
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T   # (n_samples, len(wl))
    return spectra, coeffs

def filter_response(wl, center, width):
    """Simple top‑hat filter centred at `center` with full width `width`."""
    return ((wl >= center - width/2) & (wl <= center + width/2)).astype(float)

def build_filters(wl, centers, widths):
    return [filter_response(wl, c, w) for c, w in zip(centers, widths)]

def compute_photometry(spectra, filters, wl):
    """
    Compute broadband fluxes (not magnitudes).
    flux = ∫ S(λ) R(λ) dλ / ∫ R(λ) dλ
    """
    phots = []
    for filt in filters:
        num = trapz(spectra * filt, wl, axis=1)
        den = trapz(filt, wl)
        phots.append(num / den)
    return np.column_stack(phots)

# ------------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------------
def construct_design_matrix(basis, filters, wl):
    """Matrix A such that y ≈ A c"""
    A = []
    for filt in filters:
        # project each basis onto filter
        proj = trapz(basis * filt, wl, axis=0) / trapz(filt, wl)
        A.append(proj)
    return np.vstack(A).T   # shape (n_basis, n_bands)

def reconstruct_spectra(photometry, A, wl, basis, alpha=1e-2):
    """Least‑squares estimate of coefficients then reconstruct spectra."""
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(A.T, photometry.T)           # solve for each sample
    coeffs_est = ridge.coef_.T             # (n_samples, n_basis)
    spectra_est = coeffs_est @ basis.T    # (n_samples, len(wl))
    return spectra_est, coeffs_est

# ------------------------------------------------------------------
# Main routine – synthetic experiment
# ------------------------------------------------------------------
if __name__ == "__main__":
    # wavelength grid
    wl = np.linspace(350, 950, 601)  # nm

    # Build basis
    basis = build_basis(wl, n_basis=5, seed=42)

    # Generate synthetic spectra
    spectra_true, coeffs_true = generate_spectra(100, wl, basis, seed=7)

    # Define photometric bands (e.g., U, B, V, R)
    filt_centers = [360, 440, 550, 640]          # nm
    filt_widths  = [90, 80, 100, 90]             # nm
    filters = build_filters(wl, filt_centers, filt_widths)

    # Compute photometry
    photometry = compute_photometry(spectra_true, filters, wl)

    # Construct design matrix
    A = construct_design_matrix(basis, filters, wl)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, A, wl, basis)

    # Evaluate
    rmse_spec = np.sqrt(np.mean((spectra_true - spectra_rec)**2))
    rmse_coeff = np.sqrt(np.mean((coeffs_true - coeffs_rec)**2))

    print(f"RMSE on spectra: {rmse_spec:.4f}")
    print(f"RMSE on coefficients: {rmse_coeff:.4f}")

    # Simple sanity check: plot one case (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(6,4))
        plt.plot(wl, spectra_true[idx], label="True")
        plt.plot(wl, spectra_rec[idx], '--', label="Reconstructed")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux (arb. units)")
        plt.title("Spectral Reconstruction Example")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass