import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# 1. Spectral model – Gaussian basis functions
# ------------------------------------------------------------------
def gaussian_basis(wave, centers, widths):
    """
    Create a Gaussian basis matrix.

    Parameters
    ----------
    wave : ndarray (n_wave,)
        Wavelength grid.
    centers : ndarray (n_basis,)
        Centers of the Gaussian basis.
    widths : ndarray (n_basis,)
        Widths (sigma) of the Gaussian basis.

    Returns
    -------
    basis : ndarray (n_wave, n_basis)
    """
    wave = wave[:, None]
    return np.exp(-0.5 * ((wave - centers)**2) / widths**2)


# ------------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis, rng=None):
    """
    Draw random coefficients and form spectra.

    Parameters
    ----------
    n_samples : int
        Number of spectra to generate.
    basis : ndarray (n_wave, n_basis)
    rng : np.random.Generator or None

    Returns
    -------
    spectra : ndarray (n_samples, n_wave)
    coeffs : ndarray (n_samples, n_basis)
    """
    rng = rng if rng is not None else np.random.default_rng()
    coeffs = rng.standard_normal(size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs


# ------------------------------------------------------------------
# 3. Photometry generation
# ------------------------------------------------------------------
def filter_response(wave, center, width):
    """
    Simple Gaussian filter response.
    """
    return np.exp(-0.5 * ((wave - center) ** 2) / width ** 2)


def generate_filters(wave):
    """
    Create a set of four filters (U,B,V,R).
    """
    filters = {}
    centers = {'U': 365, 'B': 445, 'V': 551, 'R': 658}
    width = 40
    for name, cen in centers.items():
        filters[name] = filter_response(wave, cen, width)
    return np.array(list(filters.values()))  # (n_filters, n_wave)


def compute_photometry(spectra, filters, wave):
    """
    Integrate each spectrum through each filter.

    Parameters
    ----------
    spectra : ndarray (n_samples, n_wave)
    filters : ndarray (n_filters, n_wave)
    wave : ndarray (n_wave,)

    Returns
    -------
    photometry : ndarray (n_samples, n_filters)
    """
    photometry = np.empty((spectra.shape[0], filters.shape[0]))
    for i in range(filters.shape[0]):
        photometry[:, i] = simps(spectra * filters[i][None, :], wave, axis=1)
    return photometry


# ------------------------------------------------------------------
# 4. Reconstruction
# ------------------------------------------------------------------
def build_design_matrix(basis, filters, wave):
    """
    Build matrix A such that photometry = A @ coeffs
    """
    n_basis = basis.shape[1]
    n_filters = filters.shape[0]
    A = np.empty((n_filters, n_basis))
    for i in range(n_filters):
        A[i] = simps(basis * filters[i][None, :], wave, axis=0)
    return A


def reconstruct_spectra(photometry, A, wave, alpha=1.0):
    """
    Reconstruct spectra from photometry using ridge regression.

    Parameters
    ----------
    photometry : ndarray (n_samples, n_filters)
    A : ndarray (n_filters, n_basis)
    wave : ndarray (n_wave,)
    alpha : float
        Ridge regularization strength.

    Returns
    -------
    recon_spectra : ndarray (n_samples, n_wave)
    coeffs_hat : ndarray (n_samples, n_basis)
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(A, photometry.T)
    coeffs_hat = ridge.predict(photometry.T).T
    recon_spectra = coeffs_hat @ basis.T
    return recon_spectra, coeffs_hat


# ------------------------------------------------------------------
# 5. Main workflow
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid
    wave = np.arange(400, 1001, 5)          # 400–1000 nm

    # Basis
    n_basis = 10
    centers = np.linspace(420, 980, n_basis)
    widths = np.full(n_basis, 50.)
    basis = gaussian_basis(wave, centers, widths)

    # Filters
    filt = generate_filters(wave)            # (n_filters, n_wave)

    # Synthetic spectra
    n_samp = 50
    spectra, true_coeffs = generate_synthetic_spectra(n_samp, basis, rng)

    # Photometry
    phot = compute_photometry(spectra, filt, wave)

    # Design matrix
    A = build_design_matrix(basis, filt, wave)

    # Reconstruction
    recon_spectra, coeffs_hat = reconstruct_spectra(phot, A, wave, alpha=0.1)

    # Error metric
    rmse = np.sqrt(np.mean((spectra - recon_spectra) ** 2))
    print(f"RMSE between original and reconstructed spectra: {rmse:.6f}")