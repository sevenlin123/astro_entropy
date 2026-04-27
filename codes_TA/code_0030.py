import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression


def create_wavelength_grid(w_min=3000, w_max=25000, n=2000):
    """Create a wavelength grid in Angstroms."""
    return np.linspace(w_min, w_max, n)


def create_gaussian_basis(n_basis, wavelength, amp_range=(0.5, 1.5), sigma_range=(200, 1500)):
    """Generate a set of Gaussian basis functions."""
    centers = np.linspace(wavelength.min(), wavelength.max(), n_basis)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_basis)
    amps = np.random.uniform(*amp_range, size=n_basis)
    basis = []
    for c, s, a in zip(centers, sigmas, amps):
        gauss = a * np.exp(-0.5 * ((wavelength - c) / s)**2)
        basis.append(gauss)
    return np.array(basis)  # shape (n_basis, len(wavelength))


def generate_synthetic_spectra(n_samples, basis):
    """Generate synthetic spectra as linear combinations of basis functions."""
    coeffs = np.random.randn(n_samples, basis.shape[0])
    spectra = coeffs @ basis  # shape (n_samples, len(wavelength))
    return spectra, coeffs


def create_tophat_filter(wavelength, f_min, f_max):
    """Create a simple top-hat filter transmission curve."""
    trans = np.zeros_like(wavelength)
    mask = (wavelength >= f_min) & (wavelength <= f_max)
    trans[mask] = 1.0
    return trans


def generate_bandpasses(n_filters, wavelength):
    """Generate a set of top-hat filters."""
    bins = np.linspace(wavelength.min(), wavelength.max(), n_filters + 1)
    filters = []
    for i in range(n_filters):
        filt = create_tophat_filter(wavelength, bins[i], bins[i+1])
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, len(wavelength))


def compute_photometry(spectra, wavelength, filters):
    """
    Compute synthetic photometric fluxes by integrating
    spectrum * filter over wavelength.
    """
    photometry = []
    for filt in filters:
        flux = simps(spectra * filt, wavelength, axis=1)
        photometry.append(flux)
    return np.vstack(photometry).T  # shape (n_samples, n_filters)


def reconstruct_coefficients(photometry, filters, basis):
    """
    Reconstruct basis coefficients from photometry via linear regression.
    The design matrix is (filters * basis) summed over wavelength.
    """
    # Build matrix X where X_{ij} = integral(filter_i * basis_j)
    X = []
    for filt in filters:
        row = simps(filt[:, None] * basis, axis=1)
        X.append(row)
    X = np.vstack(X).T  # shape (n_filters, n_basis)
    # Solve least squares: photometry = X @ coeffs.T
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, photometry)
    coeffs_rec = reg.coef_.T  # shape (n_samples, n_basis)
    return coeffs_rec


def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from recovered coefficients."""
    return coeffs @ basis  # shape (n_samples, len(wavelength))


def main():
    # 1. Set up wavelength grid and basis functions
    wavelength = create_wavelength_grid()
    basis = create_gaussian_basis(n_basis=15, wavelength=wavelength)

    # 2. Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples=50, basis=basis)

    # 3. Generate photometric data
    filters = generate_bandpasses(n_filters=5, wavelength=wavelength)
    photometry = compute_photometry(spectra_true, wavelength, filters)

    # 4. Reconstruct spectra from photometry
    coeffs_rec = reconstruct_coefficients(photometry, filters, basis)
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # 5. Compare results (simple L2 norm)
    error = np.linalg.norm(spectra_true - spectra_rec) / np.linalg.norm(spectra_true)
    print(f"Relative reconstruction error: {error:.4f}")


if __name__ == "__main__":
    main()