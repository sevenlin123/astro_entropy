import numpy as np
from scipy import stats

def create_wavelength_grid(start=300.0, end=2500.0, step=5.0):
    """Create a regular wavelength grid."""
    return np.arange(start, end + step, step)

def gaussian_basis(wavelength, centers, sigma):
    """Generate Gaussian basis functions."""
    # centers: array of center wavelengths
    # sigma: width of Gaussians
    basis = np.exp(-0.5 * ((wavelength[:, None] - centers[None, :]) / sigma)**2)
    return basis  # shape (len(wavelength), len(centers))

def generate_random_coefficients(n_samples, n_basis, rng=np.random.default_rng()):
    """Random non‑negative coefficients for synthetic spectra."""
    return rng.uniform(low=0.1, high=1.0, size=(n_samples, n_basis))

def generate_synthetic_spectra(wavelength, basis, coeffs):
    """Linear combination of basis functions."""
    return basis @ coeffs.T  # shape (len(wavelength), n_samples)

def create_filters(wavelength, filter_centers, filter_widths):
    """
    Simple rectangular filter transmission curves.
    filter_centers: list of center wavelengths
    filter_widths: list of full width at half max (same units as wavelength)
    """
    filters = []
    for center, width in zip(filter_centers, filter_widths):
        trans = np.logical_and(wavelength >= center - width / 2,
                               wavelength <= center + width / 2).astype(float)
        filters.append(trans)
    return np.array(filters)  # shape (n_filters, len(wavelength))

def compute_photometry(spectra, filters, delta_lambda):
    """
    Compute photometric fluxes by integrating spectra over filters.
    spectra: shape (len(wavelength), n_samples)
    filters: shape (n_filters, len(wavelength))
    """
    # Broadcast multiplication: (n_filters, len) x (len, n_samples) -> (n_filters, n_samples)
    integrands = filters[:, :, None] * spectra[None, :, :]
    return integrands.sum(axis=1) * delta_lambda  # shape (n_filters, n_samples)

def compute_k_matrix(basis, filters, delta_lambda):
    """
    Compute matrix K such that photometry = K @ coeffs
    basis: shape (len(wavelength), n_basis)
    filters: shape (n_filters, len(wavelength))
    """
    # Multiply filter with basis and integrate
    k = filters @ basis * delta_lambda  # shape (n_filters, n_basis)
    return k

def reconstruct_coeffs_from_photometry(photometry, K):
    """
    Solve least‑squares problem K @ coeffs = photometry
    """
    coeffs, *_ = np.linalg.lstsq(K, photometry, rcond=None)
    return coeffs  # shape (n_basis,)

def reconstruct_spectrum_from_coeffs(coeffs, basis):
    """
    Reconstruct spectrum as linear combination of basis functions.
    """
    return basis @ coeffs  # shape (len(wavelength),)

def main():
    rng = np.random.default_rng(seed=42)

    # 1. Wavelength grid
    wave = create_wavelength_grid()
    delta_lambda = wave[1] - wave[0]

    # 2. Spectral basis
    n_basis = 20
    centers = np.linspace(wave.min(), wave.max(), n_basis)
    sigma = 50.0  # width of Gaussian basis functions
    basis = gaussian_basis(wave, centers, sigma)

    # 3. Synthetic spectra
    n_samples = 5
    coeffs_true = generate_random_coefficients(n_samples, n_basis, rng=rng)
    spectra = generate_synthetic_spectra(wave, basis, coeffs_true)  # shape (len, n_samples)

    # 4. Filter definitions
    filter_centers = [450.0, 550.0, 650.0, 750.0, 850.0]  # nm
    filter_widths = [100.0] * len(filter_centers)           # nm
    filters = create_filters(wave, filter_centers, filter_widths)

    # 5. Compute photometric data
    photometry = compute_photometry(spectra, filters, delta_lambda)  # shape (n_filters, n_samples)

    # 6. Compute K matrix
    K = compute_k_matrix(basis, filters, delta_lambda)  # shape (n_filters, n_basis)

    # 7. Reconstruct the first spectrum from its photometry
    y = photometry[:, 0]            # photometric fluxes for first sample
    coeffs_rec = reconstruct_coeffs_from_photometry(y, K)
    spectrum_rec = reconstruct_spectrum_from_coeffs(coeffs_rec, basis)

    # 8. Compare original and reconstructed spectrum
    spectrum_orig = spectra[:, 0]
    rms_error = np.sqrt(np.mean((spectrum_orig - spectrum_rec)**2))
    print(f"RMS reconstruction error: {rms_error:.4e}")

if __name__ == "__main__":
    main()