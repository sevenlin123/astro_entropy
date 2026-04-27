import numpy as np
from sklearn.linear_model import Ridge

def spectral_model(params, wavelengths, basis_funcs):
    """Compute spectrum as linear combination of basis functions."""
    return np.dot(basis_funcs, params)

def get_basis_functions(wavelengths, n_basis=5):
    """Generate a set of Gaussian basis functions."""
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = np.full(n_basis, (wavelengths.max() - wavelengths.min()) / (n_basis * 4))
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :]) ** 2)
    return basis  # shape (len(wavelengths), n_basis)

def get_filter_response(wavelengths, center, width):
    """Simple Gaussian filter response."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def generate_synthetic_spectra(n_samples, wavelengths, basis_funcs):
    """Generate spectra with random coefficients."""
    n_basis = basis_funcs.shape[1]
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = np.dot(coeffs, basis_funcs.T)  # shape (n_samples, len(wavelengths))
    return spectra, coeffs

def generate_photometric_data(spectra, filters):
    """Integrate spectra over filter responses."""
    n_samples, n_wav = spectra.shape
    n_filters = len(filters)
    photometry = np.zeros((n_samples, n_filters))
    for i, filt in enumerate(filters):
        # trapezoidal integration over wavelength grid
        photometry[:, i] = np.trapz(spectra * filt[None, :], axis=1)
    return photometry

def reconstruct_coefficients(photometry, filter_matrix, alpha=1.0):
    """Reconstruct spectrum coefficients from photometry using ridge regression."""
    # filter_matrix shape: (n_filters, n_basis)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(filter_matrix, photometry.T)  # X=filter_matrix, y=photometry.T
    recon_coeffs = ridge.coef_.T  # shape (n_samples, n_basis)
    return recon_coeffs

def reconstruct_spectra(coeffs, basis_funcs):
    """Build spectra from estimated coefficients."""
    return np.dot(coeffs, basis_funcs.T)

# ---------- Example usage ----------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(3000, 8000, 1000)  # Angstrom

    # Basis functions
    basis = get_basis_functions(wav, n_basis=6)

    # Synthetic spectra
    n_samples = 10
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, wav, basis)

    # Filters (Gaussian)
    filter_centers = [4000, 5000, 6000]
    filter_widths   = [200, 200, 200]
    filters = [get_filter_response(wav, c, w) for c, w in zip(filter_centers, filter_widths)]

    # Photometry
    phot = generate_photometric_data(spectra, filters)

    # Filter matrix (basis integrated over filters)
    filter_matrix = np.array([np.trapz(basis * f[None, :], axis=0) for f in filters])  # shape (n_filters, n_basis)

    # Reconstruct coefficients
    recon_coeffs = reconstruct_coefficients(phot, filter_matrix, alpha=0.5)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectra(recon_coeffs, basis)

    # Compare true vs reconstructed spectra (simple print)
    for i in range(n_samples):
        print(f"Sample {i}:")
        print("  True coeffs   :", true_coeffs[i])
        print("  Recon coeffs  :", recon_coeffs[i])
        print("  Spectra diff  :", np.linalg.norm(spectra[i] - recon_spectra[i]))