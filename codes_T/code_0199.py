import numpy as np
from scipy.integrate import trapz


def create_basis(wavelength, n_basis):
    """
    Create a set of Gaussian basis functions sampled on the wavelength grid.
    Returns an array of shape (n_basis, len(wavelength)).
    """
    centres = np.linspace(wavelength[0], wavelength[-1], n_basis)
    widths = (wavelength[-1] - wavelength[0]) / (2 * n_basis)
    basis = np.zeros((n_basis, len(wavelength)))
    for i, c in enumerate(centres):
        basis[i] = np.exp(-0.5 * ((wavelength - c) / widths) ** 2)
    return basis


def create_filters():
    """
    Define three simple top-hat filter transmission curves.
    Each filter is represented by a dictionary with wavelength grid and response.
    """
    filters = []
    # J band 1200–1400 nm
    wl_j = np.linspace(1200, 1400, 201)
    filt_j = np.ones_like(wl_j)
    filters.append({'name': 'J', 'wl': wl_j, 'trans': filt_j})

    # H band 1500–1700 nm
    wl_h = np.linspace(1500, 1700, 201)
    filt_h = np.ones_like(wl_h)
    filters.append({'name': 'H', 'wl': wl_h, 'trans': filt_h})

    # K band 2000–2400 nm
    wl_k = np.linspace(2000, 2400, 401)
    filt_k = np.ones_like(wl_k)
    filters.append({'name': 'K', 'wl': wl_k, 'trans': filt_k})
    return filters


def compute_basis_filter_matrix(basis, wl, filters):
    """
    Compute the matrix A such that photometric flux = A @ coeffs.
    A has shape (n_filters, n_basis).
    """
    n_filters = len(filters)
    n_basis = basis.shape[0]
    A = np.zeros((n_filters, n_basis))
    for j, filt in enumerate(filters):
        # Interpolate basis onto filter grid
        interp_basis = np.interp(filt['wl'], wl, basis, axis=1)
        integrand = interp_basis * filt['trans'][:, None]
        A[j] = trapz(integrand, filt['wl'], axis=0)
    return A


def compute_photometric_fluxes(spectra, wl, filters):
    """
    Given an array of spectra (n_samples, len(wl)), compute photometric fluxes
    for each filter. Returns array of shape (n_samples, n_filters).
    """
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    fluxes = np.zeros((n_samples, n_filters))
    for j, filt in enumerate(filters):
        # Interpolate each spectrum onto filter grid
        interp_spec = np.interp(filt['wl'], wl, spectra, axis=1)
        integrand = interp_spec * filt['trans'][:, None]
        fluxes[:, j] = trapz(integrand, filt['wl'], axis=1)
    return fluxes


def reconstruct_coefficients(phot_fluxes, A):
    """
    Reconstruct coefficient vectors from photometric fluxes via least-squares.
    phot_fluxes: (n_samples, n_filters)
    A: (n_filters, n_basis)
    Returns coeffs: (n_samples, n_basis)
    """
    n_samples = phot_fluxes.shape[0]
    n_basis = A.shape[1]
    coeffs = np.zeros((n_samples, n_basis))
    for k in range(n_samples):
        coeffs[k], _, _, _ = np.linalg.lstsq(A, phot_fluxes[k], rcond=None)
    return coeffs


def reconstruct_spectra_from_coeffs(coeffs, wl, basis):
    """
    Build spectra from reconstructed coefficients.
    coeffs: (n_samples, n_basis)
    Returns spectra: (n_samples, len(wl))
    """
    return coeffs @ basis


def main():
    np.random.seed(0)

    # Wavelength grid for the underlying spectral model (400–800 nm)
    wl = np.linspace(400, 800, 81)

    # Create basis functions
    n_basis = 5
    basis = create_basis(wl, n_basis)

    # Generate synthetic spectra
    n_samples = 30
    true_coeffs = np.random.randn(n_samples, n_basis)
    spectra = true_coeffs @ basis  # (n_samples, len(wl))

    # Define filters
    filters = create_filters()

    # Compute photometric fluxes
    phot_fluxes = compute_photometric_fluxes(spectra, wl, filters)

    # Compute basis-filter integration matrix
    A = compute_basis_filter_matrix(basis, wl, filters)

    # Reconstruct coefficients from photometry
    recon_coeffs = reconstruct_coefficients(phot_fluxes, A)

    # Reconstruct spectra from coefficients
    recon_spectra = reconstruct_spectra_from_coeffs(recon_coeffs, wl, basis)

    # Evaluate reconstruction quality
    mse = np.mean((spectra - recon_spectra) ** 2)
    print(f"Mean squared reconstruction error: {mse:.4f}")


if __name__ == "__main__":
    main()