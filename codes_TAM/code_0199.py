import numpy as np
from sklearn.linear_model import Ridge


def spectral_basis(wavelengths, n_components=20):
    """Create a set of Gaussian basis functions."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_components)
    widths = np.full(n_components, 30.0)  # fixed width
    basis = np.array(
        [
            np.exp(-0.5 * ((wavelengths - c) / w) ** 2)
            for c, w in zip(centers, widths)
        ]
    )
    return basis.T  # shape (n_wavelengths, n_components)


def generate_synthetic_spectra(n_spectra, basis, rng=None):
    """Generate synthetic spectra as random linear combinations of basis."""
    rng = rng or np.random.default_rng()
    coeffs = rng.standard_normal((n_spectra, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs


def filter_transmissions(wavelengths, centers, widths):
    """Construct simple Gaussian filter responses."""
    filters = np.array(
        [
            np.exp(-0.5 * ((wavelengths - c) / w) ** 2)
            for c, w in zip(centers, widths)
        ]
    )
    return filters  # shape (n_filters, n_wavelengths)


def photometric_flux(spectra, filters):
    """Integrate spectra over filter curves to obtain photometric fluxes."""
    return spectra @ filters.T  # shape (n_spectra, n_filters)


def reconstruct_spectra(photometry, filters, basis, alpha=1.0):
    """
    Reconstruct spectra from photometry by fitting coefficients to match
    integrated filter responses. Uses Ridge regression for stability.
    """
    # Build matrix linking basis coefficients to photometric flux:
    # A[j, k] = integral of basis_k * filter_j over wavelength
    n_filters, n_wavelengths = filters.shape
    n_components = basis.shape[1]
    A = np.zeros((n_filters, n_components))
    for j in range(n_filters):
        for k in range(n_components):
            A[j, k] = np.sum(basis[:, k] * filters[j, :])

    # Fit ridge regression: photometry.T ≈ A @ coeffs.T
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, photometry.T)
    coeffs_rec = ridge.coef_.T  # shape (n_spectra, n_components)
    spectra_rec = coeffs_rec @ basis.T
    return spectra_rec, coeffs_rec


if __name__ == "__main__":
    # Define wavelength grid
    wl = np.linspace(300, 800, 500)  # nm

    # Generate basis and synthetic spectra
    basis = spectral_basis(wl, n_components=15)
    spectra, coeff_true = generate_synthetic_spectra(30, basis, rng=np.random.default_rng(42))

    # Define filter system
    filt_centers = [350, 450, 550, 650, 750]
    filt_widths = [40, 50, 60, 70, 80]
    filters = filter_transmissions(wl, filt_centers, filt_widths)

    # Generate photometric data
    phot = photometric_flux(spectra, filters)

    # Reconstruct spectra from photometry
    spectra_rec, coeff_rec = reconstruct_spectra(phot, filters, basis, alpha=0.1)

    # Evaluate reconstruction error
    error = np.linalg.norm(spectra - spectra_rec, axis=1).mean()
    print(f"Average reconstruction L2 error: {error:.4f}")