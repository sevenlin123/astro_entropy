import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

# ----------------------- Utility Functions -----------------------
def wavelength_grid(n_points=1000, wl_min=400.0, wl_max=800.0):
    """Create a linear wavelength grid."""
    return np.linspace(wl_min, wl_max, n_points)


def gaussian_basis(wl, n_basis=10, sigma=15.0):
    """
    Build an orthogonal Gaussian basis on the wavelength grid.
    Returns a matrix of shape (n_points, n_basis).
    """
    centers = np.linspace(wl.min(), wl.max(), n_basis)
    basis = np.array([norm.pdf(wl, loc=c, scale=sigma) for c in centers]).T
    # normalize each basis vector
    basis /= np.linalg.norm(basis, axis=0)
    return basis


def random_coefficients(n_samples, n_basis, scale=1.0, rng=None):
    """Generate random linear combination coefficients."""
    rng = np.random.default_rng(rng)
    return rng.normal(scale=scale, size=(n_samples, n_basis))


def synthetic_spectra(coeffs, basis):
    """
    Compute spectra as linear combinations of basis functions.
    coeffs: (n_samples, n_basis)
    basis: (n_points, n_basis)
    Returns spectra: (n_samples, n_points)
    """
    return coeffs @ basis.T


def tophat_filters(wl, n_filters=5, width=30.0):
    """
    Create simple top-hat filter responses.
    Returns filters: (n_filters, n_points)
    """
    centers = np.linspace(wl.min() + width, wl.max() - width, n_filters)
    filt = np.zeros((n_filters, wl.size))
    for i, cen in enumerate(centers):
        mask = np.abs(wl - cen) <= width / 2.0
        filt[i, mask] = 1.0
    return filt


def photometric_fluxes(spectra, filters):
    """
    Compute photometric fluxes by integrating spectra over filters.
    spectra: (n_samples, n_points)
    filters: (n_filters, n_points)
    Returns fluxes: (n_samples, n_filters)
    """
    return spectra @ filters.T


def build_design_matrix(filters, basis):
    """
    Construct the design matrix mapping basis coefficients to photometric fluxes.
    filters: (n_filters, n_points)
    basis:   (n_points, n_basis)
    Returns: (n_filters, n_basis)
    """
    return filters @ basis  # shape (n_filters, n_basis)


def reconstruct_coefficients(fluxes, design_matrix, alpha=1e-4):
    """
    Reconstruct basis coefficients from photometric fluxes using Ridge regression.
    fluxes:      (n_samples, n_filters)
    design_matrix: (n_filters, n_basis)
    alpha: regularization strength.
    Returns: coeffs: (n_samples, n_basis)
    """
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(design_matrix.T, fluxes.T)  # shape match: X (n_basis, n_filters), y (n_basis, n_samples)
    return reg.coef_.T  # shape (n_samples, n_basis)


# ----------------------- Main Execution -----------------------
if __name__ == "__main__":
    # Set up wavelength grid
    wl = wavelength_grid()

    # Build Gaussian basis functions
    basis = gaussian_basis(wl, n_basis=12, sigma=12.0)

    # Generate random coefficients for synthetic spectra
    rng = 42
    coeffs_true = random_coefficients(n_samples=50, n_basis=basis.shape[1], scale=5.0, rng=rng)

    # Compute true synthetic spectra
    spectra_true = synthetic_spectra(coeffs_true, basis)

    # Define simple top-hat filters
    filters = tophat_filters(wl, n_filters=6, width=40.0)

    # Compute photometric fluxes from true spectra
    fluxes_obs = photometric_fluxes(spectra_true, filters)

    # Build design matrix (filters * basis)
    design = build_design_matrix(filters, basis)

    # Reconstruct coefficients from photometry
    coeffs_recon = reconstruct_coefficients(fluxes_obs, design, alpha=1e-3)

    # Reconstruct spectra from estimated coefficients
    spectra_recon = synthetic_spectra(coeffs_recon, basis)

    # Simple diagnostics: RMS error between true and reconstructed spectra
    rms_error = np.sqrt(np.mean((spectra_true - spectra_recon)**2))
    print(f"Reconstruction RMS error over all spectra: {rms_error:.4f}")

    # Optionally, compare a single spectrum
    idx = 0
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(wl, spectra_true[idx], label="True Spectrum")
    plt.plot(wl, spectra_recon[idx], '--', label="Reconstructed Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (arbitrary units)")
    plt.title("Spectrum Reconstruction Example")
    plt.legend()
    plt.tight_layout()
    plt.show()