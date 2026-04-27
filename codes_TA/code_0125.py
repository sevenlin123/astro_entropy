import numpy as np
from scipy.special import erf
from sklearn.linear_model import Ridge

# ----- Spectral model -----
def create_basis(wavelengths, n_basis):
    """
    Create a set of Gaussian basis spectra.
    :param wavelengths: 1-D array of wavelengths.
    :param n_basis: Number of basis functions.
    :return: 2-D array (len(wavelengths), n_basis).
    """
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    widths = (wavelengths[-1] - wavelengths[0]) / (n_basis * 2.0)
    basis = np.exp(-((wavelengths[:, None] - centers[None, :]) ** 2) / (2 * widths**2))
    return basis

def generate_synthetic_spectrum(basis, coeffs):
    """
    Linear combination of basis spectra.
    """
    return basis @ coeffs

# ----- Photometry generation -----
def gaussian_filter(wavelengths, center, width):
    """
    Rectangular-ish Gaussian bandpass filter.
    """
    return np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))

def construct_filters(wavelengths, centers, width):
    """
    Construct filter transmission curves.
    """
    return np.array([gaussian_filter(wavelengths, c, width) for c in centers])

def filter_matrix(basis, filters):
    """
    Compute filter response matrix: M_ij = ∫ basis_j * filter_i dλ.
    """
    M = np.zeros((len(filters), basis.shape[1]))
    for i, filt in enumerate(filters):
        for j in range(basis.shape[1]):
            M[i, j] = np.trapz(basis[:, j] * filt, wavelengths)
    return M

def generate_photometry(spectra, M):
    """
    Compute photometry from spectra using filter matrix M.
    """
    coeffs = np.linalg.lstsq(M, spectra.T, rcond=None)[0]
    return (M @ coeffs).T

# ----- Spectrum reconstruction -----
def reconstruct_from_photometry(photometry, M, basis, alpha=1e-3):
    """
    Reconstruct spectrum coefficients via ridge regression.
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    ridge.fit(M, photometry.T)
    coeffs_hat = ridge.coef_.T
    reconstructed = basis @ coeffs_hat
    return reconstructed

# ----- Example usage -----
np.random.seed(42)

# Wavelength grid
wavelengths = np.linspace(400, 800, 401)  # nm

# Basis spectra
n_basis = 5
basis = create_basis(wavelengths, n_basis)

# Synthetic spectra
n_spectra = 10
true_coeffs = np.abs(np.random.randn(n_spectra, n_basis))  # positive fluxes
spectra = np.array([generate_synthetic_spectrum(basis, c) for c in true_coeffs])

# Filters
filter_centers = [450, 550, 650]
filter_width = 20.0  # nm
filters = construct_filters(wavelengths, filter_centers, filter_width)

# Filter response matrix
M = filter_matrix(basis, filters)

# Generate photometric measurements
photometry = generate_photometry(spectra, M)

# Reconstruct spectra from photometry
reconstructed_spectra = reconstruct_from_photometry(photometry, M, basis)

# ---- Simple validation ----
err = np.mean((spectra - reconstructed_spectra)**2)
print(f"Mean squared reconstruction error: {err:.4f}")