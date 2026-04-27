import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Spectral model
# ----------------------------------------------------------------------
def build_polynomial_basis(n_basis, wavelengths):
    """
    Build a simple polynomial basis up to order n_basis-1.

    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    wavelengths : ndarray
        1‑D array of wavelengths.

    Returns
    -------
    basis : ndarray, shape (n_basis, len(wavelengths))
        Basis functions evaluated on the wavelength grid.
    """
    return np.vstack([wavelengths**k for k in range(n_basis)])


# ----------------------------------------------------------------------
# 2. Synthetic spectra generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis, scale=1.0, seed=None):
    """
    Generate synthetic spectra as random linear combinations of the basis.

    Parameters
    ----------
    n_samples : int
        Number of spectra to generate.
    basis : ndarray, shape (n_basis, n_wave)
        Basis functions.
    scale : float, optional
        Standard deviation of the random coefficients.
    seed : int, optional
        Random seed.

    Returns
    -------
    spectra : ndarray, shape (n_samples, n_wave)
        Generated spectra.
    coeffs : ndarray, shape (n_samples, n_basis)
        Coefficients used to generate the spectra.
    """
    rng = np.random.default_rng(seed)
    n_basis = basis.shape[0]
    coeffs = rng.normal(scale=scale, size=(n_samples, n_basis))
    spectra = coeffs @ basis
    return spectra, coeffs


# ----------------------------------------------------------------------
# 3. Filter generation
# ----------------------------------------------------------------------
def generate_gaussian_filters(n_filters, wavelengths,
                               sigma_min=50, sigma_max=200, seed=None):
    """
    Generate a set of Gaussian filters.

    Parameters
    ----------
    n_filters : int
        Number of filters to create.
    wavelengths : ndarray
        Wavelength grid.
    sigma_min : float
        Minimum width of the Gaussian.
    sigma_max : float
        Maximum width of the Gaussian.
    seed : int, optional
        Random seed.

    Returns
    -------
    filters : ndarray, shape (n_filters, len(wavelengths))
        Normalised filter transmission curves.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), size=n_filters)
    sigmas = rng.uniform(sigma_min, sigma_max, size=n_filters)
    filt = np.exp(-0.5 * ((wavelengths[:, None] - centers)**2) / sigmas**2)
    filt /= trapz(filt, wavelengths, axis=0)  # normalise to unit area
    return filt


# ----------------------------------------------------------------------
# 4. Photometry computation
# ----------------------------------------------------------------------
def compute_photometry_from_coeffs(coeffs, basis, filters, wavelengths):
    """
    Compute photometric fluxes from known spectrum coefficients.

    Parameters
    ----------
    coeffs : ndarray, shape (n_samples, n_basis)
        Spectrum coefficients.
    basis : ndarray, shape (n_basis, n_wave)
        Basis functions.
    filters : ndarray, shape (n_filters, n_wave)
        Filter transmission curves.
    wavelengths : ndarray
        Wavelength grid.

    Returns
    -------
    photometry : ndarray, shape (n_samples, n_filters)
        Integrated fluxes for each filter.
    """
    # Pre‑compute the filter × basis integrals
    M = trapz(filters[:, None, :] * basis[None, :, :], wavelengths, axis=-1)
    # Photometry = coeffs @ M.T
    return coeffs @ M.T


# ----------------------------------------------------------------------
# 5. Reconstruction
# ----------------------------------------------------------------------
def reconstruct_coefficients(photometry, M):
    """
    Reconstruct basis coefficients from photometric data.

    Parameters
    ----------
    photometry : ndarray, shape (n_samples, n_filters)
        Observed fluxes.
    M : ndarray, shape (n_filters, n_basis)
        Integral of filters times basis functions.

    Returns
    -------
    coeffs_rec : ndarray, shape (n_samples, n_basis)
        Estimated coefficients.
    """
    lr = LinearRegression(fit_intercept=False)
    lr.fit(M.T, photometry.T)          # X = M.T, Y = photometry.T
    return lr.coef_.T                  # shape (n_samples, n_basis)


def reconstruct_spectra(coeffs_rec, basis):
    """
    Reconstruct spectra from estimated coefficients.

    Parameters
    ----------
    coeffs_rec : ndarray, shape (n_samples, n_basis)
        Estimated coefficients.
    basis : ndarray, shape (n_basis, n_wave)
        Basis functions.

    Returns
    -------
    spectra_rec : ndarray, shape (n_samples, n_wave)
        Reconstructed spectra.
    """
    return coeffs_rec @ basis


# ----------------------------------------------------------------------
# 6. Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(300, 2500, 1500)   # 300–2500 nm

    # Build basis
    n_basis = 10
    basis = build_polynomial_basis(n_basis, wav)

    # Generate synthetic spectra
    n_samp = 50
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_samp, basis, scale=1.0, seed=42
    )

    # Generate filters
    n_filt = 8
    filters = generate_gaussian_filters(
        n_filt, wav, sigma_min=80, sigma_max=200, seed=24
    )

    # Compute photometry
    phot = compute_photometry_from_coeffs(coeffs_true, basis, filters, wav)

    # Reconstruction
    M = trapz(filters[:, None, :] * basis[None, :, :], wav, axis=-1)
    coeffs_rec = reconstruct_coefficients(phot, M)
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # Accuracy check
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec)**2, axis=1))
    print(f"Mean RMSE per spectrum: {rmse.mean():.4f}")