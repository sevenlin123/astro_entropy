import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression


def create_wavelength_grid(wmin, wmax, npts):
    """Create a linear wavelength grid."""
    return np.linspace(wmin, wmax, npts)


def generate_gaussian_basis(n_basis, wavelengths):
    """Generate a set of Gaussian basis spectra."""
    rng = np.random.default_rng()
    centers = rng.uniform(np.min(wavelengths), np.max(wavelengths), n_basis)
    widths = rng.uniform((np.max(wavelengths)-np.min(wavelengths))/8,
                         (np.max(wavelengths)-np.min(wavelengths))/3,
                         n_basis)
    amplitudes = rng.uniform(0.5, 1.5, n_basis)
    basis = []
    for c, w, a in zip(centers, widths, amplitudes):
        gauss = a * np.exp(-0.5 * ((wavelengths - c)/w)**2)
        basis.append(gauss)
    return np.vstack(basis)  # shape: (n_basis, n_wave)


def synthesize_spectra(coeffs, basis_spectra):
    """Combine basis spectra with given coefficients."""
    return coeffs @ basis_spectra  # shape: (n_samples, n_wave)


def generate_filters(n_filters, wavelengths):
    """Generate simple top‑hat filters."""
    rng = np.random.default_rng()
    filters = []
    bandwidth = (np.max(wavelengths)-np.min(wavelengths))/(n_filters*2)
    for _ in range(n_filters):
        center = rng.uniform(np.min(wavelengths)+bandwidth,
                            np.max(wavelengths)-bandwidth)
        lower = center - bandwidth/2
        upper = center + bandwidth/2
        trans = np.where((wavelengths >= lower) & (wavelengths <= upper), 1.0, 0.0)
        filters.append(trans)
    return np.vstack(filters)  # shape: (n_filters, n_wave)


def compute_photometry(spectra, filters, wavelengths):
    """Integrate each spectrum over each filter."""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    photometry = np.zeros((n_samples, n_filters))
    for i in range(n_samples):
        for j in range(n_filters):
            photometry[i, j] = trapz(spectra[i] * filters[j], wavelengths)
    return photometry  # shape: (n_samples, n_filters)


def build_response_matrix(filters, basis_spectra, wavelengths):
    """Matrix mapping basis coefficients to photometric fluxes."""
    n_filters, n_wave = filters.shape
    n_basis, _ = basis_spectra.shape
    response = np.zeros((n_filters, n_basis))
    for j in range(n_filters):
        for k in range(n_basis):
            response[j, k] = trapz(basis_spectra[k] * filters[j], wavelengths)
    return response  # shape: (n_filters, n_basis)


def reconstruct_spectra(photometry, response_matrix, basis_spectra):
    """Recover spectra from photometry using linear regression."""
    n_samples = photometry.shape[0]
    reconstructed = np.zeros((n_samples, basis_spectra.shape[1]))
    for i in range(n_samples):
        reg = LinearRegression(fit_intercept=False).fit(response_matrix, photometry[i])
        coeffs_est = reg.coef_
        reconstructed[i] = coeffs_est @ basis_spectra
    return reconstructed


def main():
    # Settings
    n_wave = 1000
    wavelength = create_wavelength_grid(400, 800, n_wave)  # nm
    n_basis = 6
    n_filters = 4
    n_samples = 5
    rng = np.random.default_rng(seed=42)

    # Basis spectra
    basis = generate_gaussian_basis(n_basis, wavelength)

    # Random coefficients for synthetic spectra
    coeffs_true = rng.normal(loc=1.0, scale=0.3, size=(n_samples, n_basis))
    spectra_true = synthesize_spectra(coeffs_true, basis)

    # Filters
    filters = generate_filters(n_filters, wavelength)

    # Photometry
    photometry = compute_photometry(spectra_true, filters, wavelength)

    # Build response matrix once
    R = build_response_matrix(filters, basis, wavelength)

    # Reconstruct spectra
    spectra_rec = reconstruct_spectra(photometry, R, basis)

    # Show results
    for i in range(n_samples):
        print(f"\nSample {i+1}")
        print("True coefficients:", coeffs_true[i])
        print("Recovered coefficients:")
        reg = LinearRegression(fit_intercept=False).fit(R, photometry[i])
        print(reg.coef_)
        err = np.linalg.norm(spectra_true[i] - spectra_rec[i]) / np.linalg.norm(spectra_true[i])
        print(f"Relative reconstruction error: {err:.4f}")


if __name__ == "__main__":
    main()