import numpy as np
from sklearn.linear_model import LinearRegression


def create_spectral_basis(wavelengths, n_basis):
    """
    Create a set of Gaussian basis spectra.
    Parameters
    ----------
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
    n_basis : int
        Number of basis spectra to generate.
    Returns
    -------
    basis : ndarray, shape (n_basis, N)
        Basis spectra.
    """
    N = len(wavelengths)
    basis = np.zeros((n_basis, N))
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    widths = (wavelengths[-1] - wavelengths[0]) / (2 * n_basis)
    for i in range(n_basis):
        basis[i] = np.exp(-0.5 * ((wavelengths - centers[i]) / widths) ** 2)
    return basis


def generate_random_coeffs(n_samples, n_basis, seed=None):
    """
    Generate random coefficients for the basis spectra.
    Parameters
    ----------
    n_samples : int
        Number of synthetic spectra.
    n_basis : int
        Number of basis components.
    seed : int, optional
        Random seed.
    Returns
    -------
    coeffs : ndarray, shape (n_samples, n_basis)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=1.0, scale=0.5, size=(n_samples, n_basis))


def synthesize_spectra(coeffs, basis):
    """
    Combine basis spectra with coefficients to produce synthetic spectra.
    Parameters
    ----------
    coeffs : ndarray, shape (n_samples, n_basis)
    basis : ndarray, shape (n_basis, N_wavelength)
    Returns
    -------
    spectra : ndarray, shape (n_samples, N_wavelength)
    """
    return coeffs @ basis


def create_filters(wavelengths, n_filters):
    """
    Create simple top‑hat filter transmission curves.
    Parameters
    ----------
    wavelengths : ndarray, shape (N,)
    n_filters : int
        Number of filters.
    Returns
    -------
    filters : ndarray, shape (n_filters, N)
    """
    N = len(wavelengths)
    filters = np.zeros((n_filters, N))
    bin_edges = np.linspace(wavelengths[0], wavelengths[-1], n_filters + 1)
    for i in range(n_filters):
        mask = (wavelengths >= bin_edges[i]) & (wavelengths < bin_edges[i + 1])
        filters[i, mask] = 1.0
    # Normalize to unit area
    for i in range(n_filters):
        filters[i] /= np.trapz(filters[i], wavelengths)
    return filters


def generate_photometry(spectra, filters, wavelengths):
    """
    Compute photometric fluxes by integrating spectra over filter responses.
    Parameters
    ----------
    spectra : ndarray, shape (n_samples, N_wavelength)
    filters : ndarray, shape (n_filters, N_wavelength)
    wavelengths : ndarray, shape (N_wavelength,)
    Returns
    -------
    photometry : ndarray, shape (n_samples, n_filters)
    """
    # Integrate using trapezoidal rule
    return spectra @ (filters.T * np.diff(np.insert(wavelengths, 0, wavelengths[0])))
    

def reconstruct_spectra_from_photometry(photometry, filters, basis):
    """
    Reconstruct spectra by solving a linear regression problem.
    Parameters
    ----------
    photometry : ndarray, shape (n_samples, n_filters)
    filters : ndarray, shape (n_filters, N_wavelength)
    basis : ndarray, shape (n_basis, N_wavelength)
    Returns
    -------
    reconstructed : ndarray, shape (n_samples, N_wavelength)
    coeffs_est : ndarray, shape (n_samples, n_basis)
    """
    # The mapping from coefficients to photometry:
    # photometry = coeffs @ (basis @ filter_matrix^T)
    # Compute effective filter projections for each basis component
    filter_proj = basis @ (filters.T * np.diff(np.insert(wavelengths, 0, wavelengths[0])))
    # Fit linear model
    lr = LinearRegression(fit_intercept=False)
    lr.fit(filter_proj.T, photometry.T)
    coeffs_est = lr.coef_.T
    reconstructed = coeffs_est @ basis
    return reconstructed, coeffs_est


if __name__ == "__main__":
    # Define wavelength grid (in nm)
    wavelengths = np.arange(300, 1001, 1)

    # Spectral basis
    n_basis = 5
    basis = create_spectral_basis(wavelengths, n_basis)

    # Synthetic spectra
    n_samples = 10
    coeffs_true = generate_random_coeffs(n_samples, n_basis, seed=42)
    spectra_true = synthesize_spectra(coeffs_true, basis)

    # Filters
    n_filters = 3
    filters = create_filters(wavelengths, n_filters)

    # Photometry
    photometry = generate_photometry(spectra_true, filters, wavelengths)

    # Reconstruction
    reconstructed_spectra, coeffs_est = reconstruct_spectra_from_photometry(
        photometry, filters, basis
    )

    # Evaluate reconstruction error
    mse = np.mean((spectra_true - reconstructed_spectra) ** 2)
    print(f"Mean squared reconstruction error: {mse:.4e}")