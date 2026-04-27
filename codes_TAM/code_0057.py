import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def define_spectral_basis(n_wavelengths, n_components):
    """
    Create an orthonormal basis for spectra using discrete sine/cosine functions.
    Returns a matrix of shape (n_wavelengths, n_components).
    """
    wl = np.arange(n_wavelengths)
    basis = np.zeros((n_wavelengths, n_components))
    for k in range(1, n_components + 1):
        basis[:, k - 1] = np.sin(np.pi * k * wl / n_wavelengths)
    # Orthonormalize (simple normalization)
    basis /= np.linalg.norm(basis, axis=0)
    return basis

# ---------- Synthetic spectra ----------
def generate_synthetic_spectra(n_samples, basis, coeff_std=1.0, noise_std=0.01):
    """
    Generate synthetic spectra as linear combinations of basis functions.
    """
    n_components = basis.shape[1]
    coeffs = np.random.normal(0, coeff_std, size=(n_samples, n_components))
    spectra = coeffs @ basis.T
    spectra += np.random.normal(0, noise_std, size=spectra.shape)
    return spectra, coeffs

# ---------- Filters ----------
def create_filters(n_filters, n_wavelengths):
    """
    Generate simple boxcar filters over the wavelength grid.
    Each filter is a vector of length n_wavelengths.
    """
    filters = np.zeros((n_filters, n_wavelengths))
    band_edges = np.linspace(0, n_wavelengths, n_filters + 1, dtype=int)
    for i in range(n_filters):
        start, end = band_edges[i], band_edges[i + 1]
        filters[i, start:end] = 1.0
    return filters

# ---------- Photometry ----------
def compute_photometry(spectra, filters):
    """
    Integrate spectra through each filter to obtain photometric fluxes.
    """
    return spectra @ filters.T

# ---------- Reconstruction ----------
def reconstruct_spectrum(photon_fluxes, filters, basis):
    """
    Reconstruct spectrum coefficients from photometric fluxes via linear regression.
    """
    # Build response matrix: filters * basis
    R = filters @ basis  # shape (n_filters, n_components)
    reg = LinearRegression(fit_intercept=False).fit(R, photon_fluxes.T)
    coeffs_pred = reg.coef_.T  # shape (n_samples, n_components)
    spectra_recon = coeffs_pred @ basis.T
    return spectra_recon, coeffs_pred

# ---------- Main execution ----------
def main():
    np.random.seed(42)

    n_samples = 200
    n_wavelengths = 500   # e.g., 500 points across 300-2500 nm
    n_components = 10
    n_filters = 5

    # Define basis and filters
    basis = define_spectral_basis(n_wavelengths, n_components)
    filters = create_filters(n_filters, n_wavelengths)

    # Generate synthetic data
    spectra, coeffs_true = generate_synthetic_spectra(n_samples, basis)
    photometry = compute_photometry(spectra, filters)

    # Reconstruct spectra
    spectra_rec, coeffs_est = reconstruct_spectrum(photometry, filters, basis)

    # Evaluate reconstruction error
    recon_error = np.mean((spectra - spectra_rec) ** 2)
    print(f"Mean squared reconstruction error: {recon_error:.6f}")

if __name__ == "__main__":
    main()