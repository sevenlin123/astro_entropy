import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# 1. Spectral model – set of basis functions (e.g. Gaussians)
# ------------------------------------------------------------------
def gaussian_basis(n_basis, lam_min=4000, lam_max=7000, n_points=1000):
    """
    Return a (n_points, n_basis) matrix of Gaussian basis functions
    centered uniformly in the wavelength range.
    """
    lam = np.linspace(lam_min, lam_max, n_points)
    centers = np.linspace(lam_min, lam_max, n_basis)
    widths = (lam_max - lam_min) / (2 * n_basis)
    basis = np.exp(-((lam[:, None] - centers[None, :])**2) / (2 * widths**2))
    return lam, basis

# ------------------------------------------------------------------
# 2. Generate synthetic spectra: random linear combo of basis
# ------------------------------------------------------------------
def synthetic_spectra(n_spec, basis):
    """
    Randomly generate `n_spec` spectra as linear combinations of `basis`.
    Returns coefficient matrix (n_spec, n_basis) and spectra matrix
    (n_spec, n_wavelength).
    """
    n_basis = basis.shape[1]
    coeffs = np.random.randn(n_spec, n_basis)
    spectra = coeffs @ basis.T
    return coeffs, spectra

# ------------------------------------------------------------------
# 3. Generate photometric filter transmission curves
# ------------------------------------------------------------------
def gaussian_filter(n_filters, lam_min=4000, lam_max=7000, n_points=1000):
    """
    Return a list of filter transmission curves, each a Gaussian.
    """
    lam = np.linspace(lam_min, lam_max, n_points)
    centers = np.linspace(lam_min, lam_max, n_filters)
    widths = (lam_max - lam_min) / (5 * n_filters)
    filters = [np.exp(-((lam - c)**2) / (2 * w**2)) for c, w in zip(centers, widths)]
    return lam, filters

# ------------------------------------------------------------------
# 4. Compute synthetic photometry from spectra
# ------------------------------------------------------------------
def compute_photometry(spectra, lam, filters):
    """
    Integrate each spectrum over each filter transmission curve.
    Returns an array of shape (n_spectra, n_filters).
    """
    phot = []
    for filt in filters:
        # flux through filter = ∫ S(λ) T(λ) dλ / ∫ T(λ) dλ
        num = np.array([simps(spec * filt, lam) for spec in spectra])
        denom = simps(filt, lam)
        phot.append(num / denom)
    return np.column_stack(phot)

# ------------------------------------------------------------------
# 5. Reconstruct spectra from photometry
# ------------------------------------------------------------------
def reconstruct_from_photometry(phot, lam, basis, filters):
    """
    Solve for basis coefficients that best reproduce the given photometry.
    Uses linear regression on the forward model: coeffs -> phot.
    Returns reconstructed spectra and estimated coefficients.
    """
    # Build design matrix mapping coeffs to photometry
    n_basis = basis.shape[1]
    design = np.zeros((len(filters), n_basis))
    for i, filt in enumerate(filters):
        # For each basis function, compute its photometric response
        design[i] = np.array([simps(basis[:, j] * filt, lam) for j in range(n_basis)]) / simps(filt, lam)
    # Fit linear model (no intercept)
    lr = LinearRegression(fit_intercept=False).fit(design.T, phot.T)
    coeffs_est = lr.coef_.T  # shape (n_spectra, n_basis)
    spectra_est = coeffs_est @ basis.T
    return coeffs_est, spectra_est

# ------------------------------------------------------------------
# 6. Demonstration
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    n_basis = 5
    n_filters = 3
    n_spectra = 10

    # Create model grid
    lam, basis = gaussian_basis(n_basis)
    _, filters = gaussian_filter(n_filters, lam_min=lam.min(), lam_max=lam.max())

    # Generate synthetic spectra
    coeffs_true, spectra_true = synthetic_spectra(n_spectra, basis)

    # Compute photometry
    phot = compute_photometry(spectra_true, lam, filters)

    # Reconstruct spectra from photometry
    coeffs_est, spectra_est = reconstruct_from_photometry(phot, lam, basis, filters)

    # Simple error metric
    err = np.mean((spectra_true - spectra_est)**2)
    print(f"Mean squared reconstruction error: {err:.4f}")

    # Example plot for one spectrum (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(8, 4))
        plt.plot(lam, spectra_true[idx], label="True")
        plt.plot(lam, spectra_est[idx], '--', label="Reconstructed")
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Flux")
        plt.title(f"Spectrum #{idx}")
        plt.legend()
        plt.show()
    except ImportError:
        pass