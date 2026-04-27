import numpy as np
from scipy.optimize import nnls

# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def gaussian(x, mu, sigma):
    """Simple 1‑D Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def create_wavelength_grid(start, stop, num_points):
    """Create an evenly spaced wavelength grid."""
    return np.linspace(start, stop, num_points)


# --------------------------------------------------------------------
# Spectral basis
# --------------------------------------------------------------------
def generate_basis_gaussian(num_basis, wavelengths):
    """
    Create a set of Gaussian basis functions centred uniformly across the grid.
    Each basis is normalised to unit L2 norm.
    """
    centers = np.linspace(
        wavelengths.min() + 0.1 * (wavelengths.max() - wavelengths.min()),
        wavelengths.max() - 0.1 * (wavelengths.max() - wavelengths.min()),
        num_basis,
    )
    sigma = (wavelengths.max() - wavelengths.min()) / (2 * num_basis)

    basis = np.array([gaussian(wavelengths, c, sigma) for c in centers])
    # Normalise each basis function
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    return basis


# --------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------
def generate_filter_responses(num_filters, wavelengths):
    """
    Create Gaussian filter transmission curves.
    """
    centers = np.linspace(
        wavelengths.min() + 0.05 * (wavelengths.max() - wavelengths.min()),
        wavelengths.max() - 0.05 * (wavelengths.max() - wavelengths.min()),
        num_filters,
    )
    sigma = (wavelengths.max() - wavelengths.min()) / (10 * num_filters)

    filters = np.array([gaussian(wavelengths, c, sigma) for c in centers])
    return filters


# --------------------------------------------------------------------
# Synthetic spectra
# --------------------------------------------------------------------
def generate_synthetic_spectra(num_samples, basis):
    """
    Random linear combination of basis functions.
    Returns spectra and the true coefficients.
    """
    coeffs = np.random.rand(num_samples, basis.shape[0])  # [0,1] random coeffs
    spectra = coeffs @ basis  # (n_samples, n_wavelengths)
    return spectra, coeffs


# --------------------------------------------------------------------
# Photometry
# --------------------------------------------------------------------
def compute_photometry(spectra, filters, wavelengths):
    """
    Integrate each spectrum over each filter to obtain photometric fluxes.
    """
    delta = wavelengths[1] - wavelengths[0]
    photometry = spectra @ filters.T * delta  # (n_samples, n_filters)
    return photometry


# --------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------
def reconstruct_coefficients(photometry, filters, basis, wavelengths):
    """
    Recover the coefficients that best reproduce the photometry
    using non‑negative least squares.
    """
    delta = wavelengths[1] - wavelengths[0]
    # Forward model matrix M (filters x basis)
    M = filters @ basis.T * delta  # shape (n_filters, n_basis)

    coeffs_est = []
    for f in photometry:
        c, _ = nnls(M, f)
        coeffs_est.append(c)
    return np.vstack(coeffs_est)  # (n_samples, n_basis)


def reconstruct_spectrum(coeffs, basis):
    """Reconstruct spectra from recovered coefficients."""
    return coeffs @ basis  # (n_samples, n_wavelengths)


# --------------------------------------------------------------------
# Demonstration
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    w_start, w_stop, w_points = 400, 800, 1000  # nm
    n_basis = 5
    n_filters = 4
    n_samples = 10

    # Wavelength grid
    wavelengths = create_wavelength_grid(w_start, w_stop, w_points)

    # Build basis and filters
    basis = generate_basis_gaussian(n_basis, wavelengths)
    filters = generate_filter_responses(n_filters, wavelengths)

    # Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis)

    # Compute photometry
    photometry = compute_photometry(spectra_true, filters, wavelengths)

    # Reconstruct coefficients
    coeffs_rec = reconstruct_coefficients(photometry, filters, basis, wavelengths)

    # Reconstruct spectra
    spectra_rec = reconstruct_spectrum(coeffs_rec, basis)

    # Simple quality check
    error = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared reconstruction error: {error:.6f}")