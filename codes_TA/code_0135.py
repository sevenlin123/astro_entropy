import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import Ridge

# ---------- Model definition ----------
def generate_basis(wavelength, n_basis=5):
    """Create a set of Gaussian basis functions."""
    mu = np.linspace(wavelength[0], wavelength[-1], n_basis)
    sigma = (wavelength[-1] - wavelength[0]) / (n_basis * 4)
    basis = np.exp(-0.5 * ((wavelength[:, None] - mu[None, :]) / sigma) ** 2)
    return basis  # shape (len(wavelength), n_basis)

def synthesize_spectrum(coeffs, basis):
    """Linear combination of basis functions."""
    return coeffs @ basis.T

# ---------- Photometry ----------
def create_filters(wavelength, n_filters=5):
    """Generate simple Gaussian filters."""
    centers = np.linspace(wavelength[0], wavelength[-1], n_filters)
    sigma = (wavelength[-1] - wavelength[0]) / (n_filters * 4)
    filters = np.exp(-0.5 * ((wavelength[:, None] - centers[None, :]) / sigma) ** 2)
    return filters  # shape (len(wavelength), n_filters)

def get_photometry(spectrum, filters):
    """Integrate spectrum over each filter."""
    return trapz(spectrum[:, None] * filters, axis=0)

# ---------- Reconstruction ----------
def compute_design_matrix(basis, filters):
    """Compute matrix B where B_{kj} = ∫ basis_j * filter_k."""
    B = trapz(basis * filters[:, :, None], axis=0)  # shape (n_filters, n_basis)
    return B.T  # transpose to shape (n_basis, n_filters) for regression

def reconstruct_from_photometry(photometry, B, alpha=0.1):
    """Recover coefficients using ridge regression."""
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(B.T, photometry)      # B^T shape (n_filters, n_basis)
    coeffs = ridge.coef_           # shape (n_basis,)
    return coeffs

# ---------- Example usage ----------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(4000, 7000, 300)  # Angstrom

    # Generate basis
    basis = generate_basis(wav, n_basis=6)

    # Create filters
    filt = create_filters(wav, n_filters=7)

    # Design matrix
    B = compute_design_matrix(basis, filt)  # shape (n_basis, n_filters)

    # Generate synthetic spectrum
    true_coeffs = np.random.randn(basis.shape[1])
    spec = synthesize_spectrum(true_coeffs, basis)

    # Get photometry
    pho = get_photometry(spec, filt)

    # Reconstruct
    rec_coeffs = reconstruct_from_photometry(pho, B)
    recon_spec = synthesize_spectrum(rec_coeffs, basis)

    # Output comparison
    print("True coefficients :", true_coeffs)
    print("Recovered coeffs :", rec_coeffs)
    print("Spectrum difference rms:", np.sqrt(np.mean((spec - recon_spec)**2)))