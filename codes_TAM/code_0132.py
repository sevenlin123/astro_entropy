import numpy as np
from scipy.special import erf
from sklearn.linear_model import Ridge

# --------------------------------------
# 1. Define wavelength grid and basis spectra
# --------------------------------------
def create_wavelength_grid(start=400, end=800, num=1000):
    """Create a wavelength array (in nm)."""
    return np.linspace(start, end, num)

def create_gaussian_basis(wavelengths, num_basis=10, rng=None):
    """Generate Gaussian basis spectra."""
    rng = rng or np.random.default_rng()
    centers = rng.uniform(wavelengths[0], wavelengths[-1], size=num_basis)
    widths  = rng.uniform((wavelengths[-1]-wavelengths[0])/20,
                          (wavelengths[-1]-wavelengths[0])/10,
                          size=num_basis)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers) / widths) ** 2)
    # normalize each basis to unit area
    basis /= basis.sum(axis=0, keepdims=True)
    return basis  # shape (num_pixels, num_basis)

# --------------------------------------
# 2. Generate synthetic spectra
# --------------------------------------
def generate_synthetic_spectra(basis, num_spectra=200, rng=None):
    """Draw random linear combinations of basis spectra."""
    rng = rng or np.random.default_rng()
    coeffs = rng.normal(size=(basis.shape[1], num_spectra))
    spectra = basis @ coeffs  # shape (num_pixels, num_spectra)
    return spectra, coeffs  # return coeffs for later evaluation

# --------------------------------------
# 3. Create filter transmission curves
# --------------------------------------
def create_gaussian_filters(wavelengths, num_filters=5, rng=None):
    """Generate Gaussian filter transmission curves."""
    rng = rng or np.random.default_rng()
    centers = rng.uniform(wavelengths[0], wavelengths[-1], size=num_filters)
    widths  = rng.uniform((wavelengths[-1]-wavelengths[0])/8,
                          (wavelengths[-1]-wavelengths[0])/4,
                          size=num_filters)
    filters = np.exp(-0.5 * ((wavelengths[:, None] - centers) / widths) ** 2)
    # normalize each filter to unit area
    filters /= filters.sum(axis=0, keepdims=True)
    return filters  # shape (num_pixels, num_filters)

# --------------------------------------
# 4. Compute photometric fluxes
# --------------------------------------
def compute_photometry(spectra, filters):
    """Integrate spectra through each filter."""
    # spectra shape (num_pixels, N), filters shape (num_pixels, F)
    return filters.T @ spectra  # shape (F, N)

# --------------------------------------
# 5. Reconstruction from photometry
# --------------------------------------
def reconstruct_spectrum_from_photometry(photometry, filters, basis, alpha=1e-3):
    """
    Reconstruct spectra from photometry using ridge regression.
    photometry: (F, N)
    filters:   (num_pixels, F)
    basis:     (num_pixels, B)
    """
    # Build design matrix that maps basis coefficients to photometric fluxes
    # M_{f,b} = integral (filter_f * basis_b)
    M = filters.T @ basis  # shape (F, B)
    # Fit ridge regression: coeffs = (M^T M + alpha I)^-1 M^T y
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(M, photometry.T)   # regress coefficients for each spectrum
    coeffs_pred = reg.coef_.T   # shape (B, N)
    # Reconstruct spectra
    spectra_recon = basis @ coeffs_pred  # shape (num_pixels, N)
    return spectra_recon, coeffs_pred

# --------------------------------------
# 6. Demo
# --------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid
    wav = create_wavelength_grid()

    # Basis spectra
    basis = create_gaussian_basis(wav, num_basis=15, rng=rng)

    # Synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(basis, num_spectra=50, rng=rng)

    # Filters
    filters = create_gaussian_filters(wav, num_filters=7, rng=rng)

    # Photometry
    photometry = compute_photometry(spectra_true, filters)

    # Reconstruction
    spectra_rec, coeffs_rec = reconstruct_spectrum_from_photometry(
        photometry, filters, basis, alpha=1e-3
    )

    # Evaluate: simple RMS error between true and reconstructed spectra
    rms_error = np.sqrt(((spectra_true - spectra_rec) ** 2).mean(axis=0))
    print("RMS error per spectrum:", rms_error.mean())