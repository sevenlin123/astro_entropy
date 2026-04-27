import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

# -------------------- Spectral model --------------------
def create_basis_functions(n_basis, wl):
    """Create Gaussian basis functions over wavelengths."""
    centers = np.linspace(wl[0], wl[-1], n_basis)
    widths = (wl[-1] - wl[0]) / (n_basis * 4)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wl - c)/widths)**2)
        basis.append(g)
    return np.vstack(basis)  # shape (n_basis, n_wl)

def synthesize_spectra(coeffs, basis):
    """Linear combination of basis functions."""
    return coeffs @ basis  # shape (n_samples, n_wl)

# -------------------- Photometry --------------------
def create_filters(n_filters, wl):
    """Gaussian bandpasses."""
    centers = np.linspace(wl[0], wl[-1], n_filters)
    widths = (wl[-1] - wl[0]) / (n_filters * 2)
    filt = []
    for c in centers:
        f = np.exp(-0.5 * ((wl - c)/widths)**2)
        filt.append(f)
    return np.vstack(filt)  # shape (n_filters, n_wl)

def compute_photometry(spectra, filters):
    """Integrate spectra over filters."""
    return spectra @ filters.T  # shape (n_samples, n_filters)

# -------------------- Reconstruction --------------------
def reconstruct_spectra(photometry, filters, alpha=1.0):
    """Reconstruct spectra from photometry using ridge regression."""
    n_samples, n_filters = photometry.shape
    n_wl = filters.shape[1]
    # Build design matrix: photometry * filter matrix
    X = photometry[:, :, None] * filters[None, :, :]  # shape (n_samples, n_filters, n_wl)
    X = X.reshape(n_samples, -1)  # flatten filters * wl
    y = np.arange(n_wl)  # dummy target? actually we want to reconstruct each wavelength
    # Instead perform multivariate ridge regression
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, spectra)
    return model.predict(X)  # shape (n_samples, n_wl)

# -------------------- Main --------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Wavelength grid
    wl = np.linspace(400, 800, 401)  # nm

    # Generate synthetic spectra
    n_samples = 200
    n_basis = 10
    basis = create_basis_functions(n_basis, wl)
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = synthesize_spectra(coeffs, basis)

    # Create filters and compute photometry
    n_filters = 5
    filters = create_filters(n_filters, wl)
    photometry = compute_photometry(spectra, filters)

    # Reconstruct spectra
    X = photometry[:, :, None] * filters[None, :, :]
    X = X.reshape(n_samples, -1)
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X, spectra)
    spectra_recon = model.predict(X)

    # Evaluate reconstruction
    mse = np.mean((spectra - spectra_recon)**2)
    print(f"Reconstruction MSE: {mse:.4f}")

    # Display an example
    import matplotlib.pyplot as plt
    idx = 0
    plt.figure(figsize=(8,4))
    plt.plot(wl, spectra[idx], label="Original")
    plt.plot(wl, spectra_recon[idx], '--', label="Reconstructed")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux")
    plt.legend()
    plt.tight_layout()
    plt.show()