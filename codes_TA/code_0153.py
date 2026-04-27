import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

def create_spectral_basis(wl, n_basis):
    """Create orthogonal Gaussian basis functions."""
    centers = np.linspace(wl.min(), wl.max(), n_basis)
    widths = (wl.max() - wl.min()) / (2 * n_basis)
    basis = []
    for c in centers:
        gauss = np.exp(-0.5 * ((wl - c) / widths)**2)
        basis.append(gauss)
    return np.vstack(basis).T  # shape (len(wl), n_basis)

def generate_synthetic_spectra(basis, n_samples, noise_level=0.01):
    """Generate synthetic spectra as random linear combinations of basis."""
    coeffs = np.random.randn(n_samples, basis.shape[1])
    spectra = coeffs @ basis.T
    spectra += noise_level * np.random.randn(*spectra.shape)
    return spectra, coeffs

def create_filters(wl, n_filters):
    """Create simple Gaussian filters."""
    centers = np.linspace(wl.min(), wl.max(), n_filters)
    widths = (wl.max() - wl.min()) / (4 * n_filters)
    filters = []
    for c in centers:
        filt = np.exp(-0.5 * ((wl - c) / widths)**2)
        filters.append(filt)
    return np.vstack(filters).T  # shape (len(wl), n_filters)

def compute_photometry(spectra, filters, wl):
    """Integrate spectra over each filter to obtain photometric fluxes."""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[1]
    phot = np.zeros((n_samples, n_filters))
    for i in range(n_samples):
        for j in range(n_filters):
            integrand = spectra[i] * filters[:, j]
            phot[i, j] = simps(integrand, wl)
    return phot

def reconstruct_spectrum(photometry, filters, basis, wl):
    """
    Reconstruct spectra from photometry by solving linear least squares
    for the basis coefficients.
    """
    n_samples, n_filters = photometry.shape
    n_basis = basis.shape[1]

    # Build design matrix M where M[j, k] = integral of basis_k * filter_j
    M = np.zeros((n_filters, n_basis))
    for j in range(n_filters):
        for k in range(n_basis):
            integrand = basis[:, k] * filters[:, j]
            M[j, k] = simps(integrand, wl)

    # Solve for coefficients for each sample
    recon_spectra = np.zeros((n_samples, len(wl)))
    for i in range(n_samples):
        coeff = LinearRegression(fit_intercept=False).fit(M, photometry[i]).coef_
        recon_spectra[i] = coeff @ basis.T
    return recon_spectra

if __name__ == "__main__":
    np.random.seed(42)

    # Wavelength grid
    wl = np.linspace(4000, 8000, 1000)  # Angstroms

    # Create basis, filters
    n_basis = 20
    basis = create_spectral_basis(wl, n_basis)

    n_filters = 5
    filters = create_filters(wl, n_filters)

    # Generate synthetic spectra
    n_samples = 10
    spectra, true_coeffs = generate_synthetic_spectra(basis, n_samples)

    # Compute photometry
    photometry = compute_photometry(spectra, filters, wl)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectrum(photometry, filters, basis, wl)

    # Simple comparison for first sample
    print("True coefficients (first sample):", true_coeffs[0])
    print("Reconstructed coefficients (first sample):",
          LinearRegression(fit_intercept=False).fit(
              np.array([simps(basis[:, k] * filters[:, j], wl)
                        for j in range(n_filters) for k in range(n_basis)]
                       ).reshape(n_filters, n_basis),
              photometry[0]).coef_)
    print("Difference between true and reconstructed spectra:",
          np.linalg.norm(spectra[0] - recon_spectra[0]))