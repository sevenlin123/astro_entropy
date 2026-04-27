import numpy as np
from scipy.special import erfc
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def gaussian_basis(n_basis, wavelengths, widths=200.0):
    """Return a matrix of Gaussian basis functions."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths) ** 2)
    return basis

# ---------- Synthetic spectra ----------
def generate_synthetic_spectra(n_samples, basis):
    """Generate spectra as random linear combinations of basis functions."""
    coeffs = np.random.randn(n_samples, basis.shape[1])
    spectra = coeffs @ basis
    return spectra, coeffs

# ---------- Filters ----------
def gaussian_filter(lambda0, width, wavelengths):
    """Return a single Gaussian filter transmission."""
    return np.exp(-0.5 * ((wavelengths - lambda0) / width) ** 2)

def make_filters(n_filters, wavelengths, widths=300.0):
    """Create a set of Gaussian filters."""
    lambdas = np.linspace(wavelengths[0] + widths, wavelengths[-1] - widths, n_filters)
    filters = np.vstack([gaussian_filter(l0, widths, wavelengths) for l0 in lambdas])
    return filters

# ---------- Photometry ----------
def compute_photometry(spectra, filters):
    """Integrate spectra over each filter to get photometric measurements."""
    # Normalize filters so their integrals equal 1
    norm_filters = filters / np.sum(filters, axis=1, keepdims=True)
    return spectra @ norm_filters.T

# ---------- Reconstruction ----------
def reconstruct_from_photometry(photometry, basis, filters):
    """Recover a spectrum from photometric data using least‑squares."""
    # Build response matrix: integral of each basis function over each filter
    R = basis.T @ filters.T          # shape (n_basis, n_filters)
    R = R.T                           # shape (n_filters, n_basis)
    # Solve for coefficients
    lr = LinearRegression(fit_intercept=False)
    lr.fit(R, photometry)
    coeffs_est = lr.coef_.T            # shape (n_basis, n_samples)
    spectra_est = coeffs_est.T @ basis
    return spectra_est, coeffs_est.T

# ---------- Example workflow ----------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(4000, 7000, 301)   # Angstroms

    # Build basis
    nbasis = 20
    basis = gaussian_basis(nbasis, wav, widths=150.0)

    # Generate synthetic spectra
    nsamples = 5
    spectra_true, coeffs_true = generate_synthetic_spectra(nsamples, basis)

    # Create filters
    nfilters = 7
    filters = make_filters(nfilters, wav, widths=250.0)

    # Compute photometry
    photometry = compute_photometry(spectra_true, filters)

    # Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_from_photometry(photometry, basis, filters)

    # Compare (print RMS errors)
    rms_true = np.sqrt(np.mean((spectra_true - spectra_rec) ** 2, axis=1))
    print("RMS error per spectrum:", rms_true)