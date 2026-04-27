import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------
# Spectral model
# ------------------------------------------------------------
def create_gaussian_basis(wavelengths, n_basis):
    """Create n_basis Gaussian basis functions over wavelengths."""
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    width = (wavelengths.max() - wavelengths.min()) / (4 * n_basis)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / width) ** 2)
    return basis  # shape (len(wavelengths), n_basis)

# ------------------------------------------------------------
# Synthetic data generation
# ------------------------------------------------------------
def generate_synthetic_spectra(basis, n_objects, noise=0.02):
    """Generate random spectra as linear combos of basis functions."""
    coeffs = np.random.randn(n_objects, basis.shape[1])
    spectra = coeffs @ basis.T
    spectra += noise * np.random.randn(*spectra.shape)
    return spectra, coeffs

def create_gaussian_filters(wavelengths, n_filters):
    """Create n_filters Gaussian filter transmission curves."""
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_filters)
    width = (wavelengths.max() - wavelengths.min()) / (6 * n_filters)
    filters = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / width) ** 2)
    # normalize each filter by its integral
    for i in range(n_filters):
        area = simps(filters[:, i], wavelengths)
        filters[:, i] /= area if area != 0 else 1.0
    return filters  # shape (len(wavelengths), n_filters)

def compute_photometry(spectra, filters, wavelengths):
    """Integrate spectra through filters to obtain photometric fluxes."""
    n_objects = spectra.shape[0]
    n_filters = filters.shape[1]
    phot = np.zeros((n_objects, n_filters))
    for i in range(n_filters):
        # product of each spectrum with filter i
        prod = spectra * filters[:, i][None, :]
        phot[:, i] = simps(prod, wavelengths, axis=1)
    return phot

# ------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------
def reconstruct_spectra(photometry, filters, basis, wavelengths, alpha=1e-4):
    """
    Reconstruct spectra from photometry using basis functions.
    Returns reconstructed spectra and fitted coefficients.
    """
    # Build design matrix: for each filter, integrate each basis * filter
    n_filters = filters.shape[1]
    n_basis = basis.shape[1]
    A = np.zeros((n_filters, n_basis))
    for j in range(n_filters):
        for k in range(n_basis):
            prod = basis[:, k] * filters[:, j]
            A[j, k] = simps(prod, wavelengths)
    # Fit coefficients using Ridge regression (avoid overfitting)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(A.T, photometry.T)  # transpose to match shapes
    coeffs_hat = model.coef_.T  # shape (n_objects, n_basis)
    recon = coeffs_hat @ basis.T
    return recon, coeffs_hat

# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (nm)
    wav = np.linspace(400, 800, 1000)

    # Create basis and generate spectra
    nbasis = 10
    basis = create_gaussian_basis(wav, nbasis)
    n_objs = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(basis, n_objs, noise=0.05)

    # Create filters and compute photometry
    nfilters = 5
    filt = create_gaussian_filters(wav, nfilters)
    phot = compute_photometry(spectra_true, filt, wav)

    # Reconstruct spectra
    spectra_rec, coeffs_est = reconstruct_spectra(phot, filt, basis, wav, alpha=1e-3)

    # Evaluate reconstruction quality
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Reconstruction MSE per wavelength: {mse:.4e}")

    # Compare true vs estimated coefficients for first object
    print("\nTrue coefficients (first object):")
    print(coeffs_true[0])
    print("Estimated coefficients (first object):")
    print(coeffs_est[0])