import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# Spectral model: a set of Gaussian basis functions
# ------------------------------------------------------------
def gaussian_basis(wavelengths, n_bases=5):
    """
    Return a matrix of basis functions evaluated at `wavelengths`.
    Each column is a Gaussian centered at evenly spaced positions.
    """
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_bases)
    widths = (wavelengths.max() - wavelengths.min()) / (n_bases * 2)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths) ** 2)
    return basis


# ------------------------------------------------------------
# Generate synthetic spectra
# ------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths, basis, rng=None):
    """
    Sample random coefficients for each spectrum and construct spectra.
    """
    rng = rng or np.random.default_rng()
    coeffs = rng.normal(size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs


# ------------------------------------------------------------
# Define photometric filters as arbitrary Gaussians
# ------------------------------------------------------------
def generate_random_filters(n_filters, wavelengths, rng=None):
    """
    Create `n_filters` Gaussian transmission curves.
    """
    rng = rng or np.random.default_rng()
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wavelengths.min(), wavelengths.max())
        width = rng.uniform(10, 30)
        trans = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
        trans /= trans.max()
        filters.append(trans)
    return np.array(filters)


# ------------------------------------------------------------
# Compute photometry (integrated flux through each filter)
# ------------------------------------------------------------
def compute_photometry(spectra, filters, wavelengths):
    """
    Integrate each spectrum over each filter.
    """
    # Interpolate spectra onto wavelengths if needed (they already match)
    phot = spectra @ filters.T
    return phot


# ------------------------------------------------------------
# Reconstruct spectrum via ridge regression (least squares with Tikhonov)
# ------------------------------------------------------------
def reconstruct_spectrum(photometry, filters, basis, alpha=0.1):
    """
    For each sample, fit basis coefficients that reproduce the photometry.
    """
    n_samples = photometry.shape[0]
    coef_est = np.empty((n_samples, basis.shape[1]))
    reg = Ridge(alpha=alpha, fit_intercept=False)
    X = filters.T @ basis  # design matrix mapping coefficients to photometry
    reg.fit(X, photometry.T)
    coef_est = reg.coef_.T
    reconstructed = coef_est @ basis.T
    return reconstructed, coef_est


# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # Wavelength grid
    wavelengths = np.linspace(400, 800, 401)  # 400–800 nm, 1 nm steps

    # Basis functions
    basis = gaussian_basis(wavelengths, n_bases=5)

    # Synthetic spectra
    n_samples = 50
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, wavelengths, basis, rng=rng)

    # Photometric filters
    n_filters = 8
    filters = generate_random_filters(n_filters, wavelengths, rng=rng)

    # Photometry
    photometry = compute_photometry(spectra, filters, wavelengths)

    # Reconstruction
    reconstructed, est_coeffs = reconstruct_spectrum(photometry, filters, basis, alpha=0.05)

    # Evaluation
    mse = np.mean((spectra - reconstructed) ** 2)
    print(f"Reconstruction MSE per wavelength: {mse:.4e}")

    # Optional: compare a single spectrum visually (requires matplotlib, omitted here)
    # print("Done.")


if __name__ == "__main__":
    main()