import numpy as np
from sklearn.linear_model import Ridge


def basis_functions(wavelengths, n_basis=5):
    """Simple polynomial basis up to degree n_basis-1."""
    return np.vstack(
        [wavelengths ** i for i in range(n_basis)]
    ).T  # shape (N_WL, n_basis)


def gaussian_filter(center, width, wavelengths):
    """Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def generate_filters(n_filters, wavelengths):
    """Generate a set of Gaussian filters."""
    centers = [360, 440, 550, 700]  # nm
    widths = [50, 50, 50, 50]       # nm
    filters = []
    for i in range(n_filters):
        filters.append(gaussian_filter(centers[i], widths[i], wavelengths))
    return np.array(filters)  # shape (n_filters, N_WL)


def generate_synthetic_spectra(n_samples, basis):
    """Generate synthetic spectra and underlying coefficients."""
    coeffs = np.random.randn(n_samples, basis.shape[1])  # shape (N_SAMP, n_basis)
    spectra = coeffs @ basis.T                          # shape (N_SAMP, N_WL)
    return spectra, coeffs


def compute_photometry(spectra, filters, wavelengths):
    """Integrate spectra through each filter to obtain photometric fluxes."""
    # spectra: (N_SAMP, N_WL)
    # filters: (N_FILT, N_WL)
    # Expand dims for broadcasting
    integrand = spectra[:, None, :] * filters[None, :, :]  # (N_SAMP, N_FILT, N_WL)
    photometry = np.trapz(integrand, wavelengths, axis=-1)  # (N_SAMP, N_FILT)
    return photometry


def reconstruct_spectra(photometry, filters, basis, wavelengths, true_coeffs):
    """
    Reconstruct spectra from photometry using a Ridge regression model
    that maps photometry -> spectral coefficients.
    """
    ridge = Ridge(alpha=1e-3, fit_intercept=False)
    ridge.fit(photometry, true_coeffs)
    coeffs_pred = ridge.predict(photometry)
    spectra_pred = coeffs_pred @ basis.T
    return spectra_pred, coeffs_pred


def main():
    np.random.seed(42)

    # Wavelength grid
    wl_min, wl_max, n_wl = 300, 800, 1000
    wavelengths = np.linspace(wl_min, wl_max, n_wl)

    # Spectral basis
    basis = basis_functions(wavelengths, n_basis=5)

    # Generate synthetic spectra
    n_samples = 200
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis)

    # Filters
    n_filters = 4
    filters = generate_filters(n_filters, wavelengths)

    # Photometric measurements
    photometry = compute_photometry(spectra_true, filters, wavelengths)

    # Reconstruction
    spectra_recon, coeffs_recon = reconstruct_spectra(
        photometry, filters, basis, wavelengths, coeffs_true
    )

    # Error metrics
    mse = np.mean((spectra_true - spectra_recon) ** 2)
    rmse = np.sqrt(mse)
    print(f"Spectral reconstruction RMSE: {rmse:.4f}")

    # Sample comparison for first 5 spectra
    for i in range(5):
        diff = np.abs(spectra_true[i] - spectra_recon[i])
        max_diff = diff.max()
        print(f"Sample {i}: max abs error in spectrum = {max_diff:.3f}")


if __name__ == "__main__":
    main()