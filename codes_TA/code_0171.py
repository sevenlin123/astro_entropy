import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

def gaussian_basis(wavelengths, centers, widths):
    """Return a matrix of Gaussian basis functions."""
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :])**2)
    return basis

def generate_synthetic_spectra(n_samples, wavelengths, basis):
    """Generate synthetic spectra as random linear combinations of basis functions."""
    n_basis = basis.shape[1]
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = coeffs @ basis.T  # shape (n_samples, n_wavelengths)
    return spectra, coeffs

def generate_filters(wavelengths, n_filters=3):
    """Generate simple Gaussian filter transmission curves."""
    rng = np.random.default_rng()
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), size=n_filters)
    widths = rng.uniform((wavelengths.max()-wavelengths.min())/20,
                         (wavelengths.max()-wavelengths.min())/10,
                         size=n_filters)
    filters = gaussian_basis(wavelengths, centers, widths)
    # normalize each filter to unit area
    filters /= np.trapz(filters, wavelengths, axis=1)[:, None]
    return filters

def compute_photometry(spectra, filters, wavelengths):
    """Integrate spectra over filters to obtain photometric measurements."""
    # Use trapezoidal integration
    integrals = np.trapz(spectra[:, :, None] * filters[None, :, :], wavelengths, axis=1)
    return integrals  # shape (n_samples, n_filters)

def reconstruct_spectra(photometry, filters, basis):
    """Reconstruct spectra from photometric data using linear regression."""
    # Build design matrix A = filters @ basis
    A = filters @ basis  # shape (n_filters, n_basis)
    # Solve for coefficients that map A -> photometry
    ridge = Ridge(alpha=1e-3, fit_intercept=False)
    ridge.fit(A, photometry.T)
    coeffs_hat = ridge.coef_.T   # shape (n_basis, n_samples)
    # Reconstruct spectra
    spectra_hat = coeffs_hat.T @ basis  # shape (n_samples, n_wavelengths)
    return spectra_hat, coeffs_hat

def main():
    # Wavelength grid
    wavelengths = np.linspace(400, 800, 200)  # nm
    # Basis functions
    centers = np.linspace(420, 780, 5)
    widths = np.full(5, 30.)
    basis = gaussian_basis(wavelengths, centers, widths)
    # Synthetic spectra
    n_samples = 20
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, wavelengths, basis)
    # Filters
    filters = generate_filters(wavelengths, n_filters=4)
    # Photometry
    photometry = compute_photometry(spectra_true, filters, wavelengths)
    # Reconstruction
    spectra_recon, coeffs_est = reconstruct_spectra(photometry, filters, basis)
    # Error
    mse = np.mean((spectra_true - spectra_recon)**2)
    print(f"Mean squared reconstruction error: {mse:.4e}")

if __name__ == "__main__":
    main()