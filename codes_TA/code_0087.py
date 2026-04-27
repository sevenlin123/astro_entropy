import numpy as np
from sklearn.linear_model import Ridge

# --- basic utilities ---------------------------------------------------------
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def create_wavelength_grid(start=400.0, end=800.0, n=1000):
    """Create a linearly spaced wavelength array (nm)."""
    return np.linspace(start, end, n)

# --- spectral model ----------------------------------------------------------
def generate_basis_functions(wl, n_basis=5):
    """Return an array of basis functions (n_basis, len(wl))."""
    centers = np.linspace(wl.min(), wl.max(), n_basis)
    sigma = (wl.max() - wl.min()) / (2 * n_basis)
    basis = np.vstack([gaussian(wl, c, sigma) for c in centers])
    return basis

def generate_synthetic_spectra(basis, n_samples=50, rng=None):
    """Generate random spectra as linear combinations of basis functions."""
    rng = np.random.default_rng(rng)
    coeffs = rng.uniform(0, 1, size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis  # shape (n_samples, n_wavelength)
    return spectra, coeffs

# --- photometric model -------------------------------------------------------
def generate_filter_responses(wl, n_filters=3):
    """Return an array of filter transmission curves (n_filters, len(wl))."""
    centers = np.linspace(wl.min(), wl.max(), n_filters)
    sigma = (wl.max() - wl.min()) / (4 * n_filters)
    filters = np.vstack([gaussian(wl, c, sigma) for c in centers])
    # Normalize to unit area
    area = np.trapz(filters, wl, axis=1)
    filters = filters / area[:, None]
    return filters

def compute_photometry(spectra, filters, wl):
    """
    Compute photometric measurements by integrating the product of each
    spectrum with each filter transmission curve.
    """
    dw = np.diff(wl, prepend=wl[0])          # integration weights
    integrands = spectra[:, :, None] * filters[None, :, :]
    # integrand shape: (n_spectra, n_filters, n_wavelength)
    phot = np.sum(integrands * dw, axis=2)    # shape (n_spectra, n_filters)
    return phot

# --- reconstruction framework -----------------------------------------------
def reconstruct_coefficients(photometry, basis, filters, wl, alpha=1e-3):
    """
    Estimate the coefficients of the basis functions that produced the
    photometric measurements. Uses ridge regression for numerical stability.
    """
    # Precompute the projection matrix M[j, k] = <basis_j, filter_k>
    n_basis = basis.shape[0]
    n_filters = filters.shape[0]
    M = np.empty((n_basis, n_filters))
    for j in range(n_basis):
        for k in range(n_filters):
            M[j, k] = np.trapz(basis[j] * filters[k], wl)
    # Solve for coefficients: photometry = coeffs @ M
    # Using ridge regression on the transpose problem
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(M.T, photometry.T)
    coeffs_est = ridge.coef_.T
    return coeffs_est

def reconstruct_spectra(coeffs_est, basis):
    """Reconstruct spectra from estimated coefficients."""
    return coeffs_est @ basis

# --- demo --------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Wavelength grid
    wl = create_wavelength_grid()

    # 2. Basis functions
    basis = generate_basis_functions(wl, n_basis=5)

    # 3. Synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(basis, n_samples=50, rng=42)

    # 4. Filters
    filters = generate_filter_responses(wl, n_filters=3)

    # 5. Photometry
    phot = compute_photometry(spectra_true, filters, wl)

    # 6. Reconstruction
    coeffs_rec = reconstruct_coefficients(phot, basis, filters, wl, alpha=1e-3)
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # 7. Simple evaluation
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec) ** 2))
    print(f"RMSE between true and reconstructed spectra: {rmse:.3f}")

    # Optional: plot one example (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        i = 0  # first spectrum
        plt.figure(figsize=(8, 4))
        plt.plot(wl, spectra_true[i], label="True")
        plt.plot(wl, spectra_rec[i], '--', label="Reconstructed")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux (arb. units)")
        plt.title("Spectral Reconstruction Example")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass