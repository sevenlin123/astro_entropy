import numpy as np
from scipy.signal import convolve
from sklearn.linear_model import Lasso

# ---------- Spectral model ----------
def define_spectral_model(n_wl=1000, n_basis=10, rng=None):
    """Create a simple basis (Gaussian bumps) for spectra."""
    rng = rng or np.random.default_rng()
    wl = np.linspace(400, 800, n_wl)  # nm
    centers = np.linspace(450, 750, n_basis)
    widths = 20 * np.ones(n_basis)
    basis = np.array([np.exp(-0.5 * ((wl - c)/w)**2) for c, w in zip(centers, widths)])
    return wl, basis.T  # shape (n_wl, n_basis)

# ---------- Synthetic spectra ----------
def generate_synthetic_spectra(n_samples, basis, rng=None):
    """Generate random spectra as linear combos of the basis."""
    rng = rng or np.random.default_rng()
    coeffs = rng.uniform(-1, 1, size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T  # shape (n_samples, n_wl)
    return coeffs, spectra

# ---------- Filters ----------
def generate_filters(n_filters, wl, rng=None):
    """Create simple Gaussian filter responses."""
    rng = rng or np.random.default_rng()
    centers = rng.uniform(wl.min(), wl.max(), size=n_filters)
    widths = rng.uniform(30, 70, size=n_filters)
    filters = np.array([np.exp(-0.5 * ((wl - c)/w)**2) for c, w in zip(centers, widths)])
    return filters  # shape (n_filters, n_wl)

# ---------- Photometry ----------
def generate_photometry(spectra, filters):
    """Integrate spectra through filters to get photometric fluxes."""
    # Normalize filters to unit integral
    filt_norm = filters / np.sum(filters, axis=1, keepdims=True)
    phot = spectra @ filt_norm.T  # shape (n_samples, n_filters)
    return phot

# ---------- Reconstruction ----------
def reconstruct_spectrum(phot, filters, basis, alpha=0.0):
    """
    Reconstruct spectrum from photometry.
    Solve for coefficients using linear regression or Lasso.
    """
    # Build response matrix: A[j,k] = ∫ T_j * basis_k
    A = filters @ basis  # shape (n_filters, n_basis)
    if alpha == 0.0:
        coeffs, *_ = np.linalg.lstsq(A, phot.T, rcond=None)
    else:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(A, phot.T)
        coeffs = lasso.coef_
    reconstructed = coeffs.T @ basis  # shape (n_samples, n_wl)
    return coeffs, reconstructed

# ---------- Main execution ----------
def main():
    rng = np.random.default_rng(42)

    # Define model
    wl, basis = define_spectral_model(n_wl=1000, n_basis=15, rng=rng)

    # Generate synthetic spectra
    n_samples = 50
    true_coeffs, spectra = generate_synthetic_spectra(n_samples, basis, rng=rng)

    # Generate filters
    n_filters = 8
    filters = generate_filters(n_filters, wl, rng=rng)

    # Get photometry
    phot = generate_photometry(spectra, filters)

    # Reconstruct spectra
    est_coeffs, recon_spectra = reconstruct_spectrum(phot, filters, basis, alpha=0.01)

    # Compare
    print("True coeffs shape:", true_coeffs.shape)
    print("Estimated coeffs shape:", est_coeffs.shape)
    print("Reconstructed spectra shape:", recon_spectra.shape)

    # Simple error metric
    err = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared reconstruction error: {err:.4e}")

if __name__ == "__main__":
    main()