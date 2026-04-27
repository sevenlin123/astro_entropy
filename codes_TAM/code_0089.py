import numpy as np
from numpy.linalg import lstsq
from scipy.stats import norm

# ----------------------------------------------------------------------
# Spectral model
# ----------------------------------------------------------------------
def create_wavelength_grid(n_pixels=1000, lam_min=400.0, lam_max=800.0):
    """Generate a wavelength grid in nm."""
    return np.linspace(lam_min, lam_max, n_pixels)


def create_gaussian_basis(wavelengths, n_basis=20):
    """
    Create a set of Gaussian basis functions across the wavelength grid.
    Returns a matrix of shape (n_pixels, n_basis).
    """
    n_pixels = wavelengths.size
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = (wavelengths.max() - wavelengths.min()) / (n_basis * 4.0)
    basis = np.zeros((n_pixels, n_basis))
    for i, (c, w) in enumerate(zip(centers, [widths]*n_basis)):
        basis[:, i] = norm.pdf(wavelengths, loc=c, scale=w)
    # normalize each basis vector
    basis /= basis.sum(axis=0, keepdims=True)
    return basis


# ----------------------------------------------------------------------
# Synthetic spectra generation
# ----------------------------------------------------------------------
def generate_random_coefficients(n_spectra, n_basis):
    """
    Generate random non‑negative coefficients for each spectrum.
    Shape: (n_spectra, n_basis)
    """
    rng = np.random.default_rng()
    coeffs = rng.normal(loc=1.0, scale=0.5, size=(n_spectra, n_basis))
    coeffs[coeffs < 0] = 0.0
    return coeffs


def synthesize_spectra(basis, coeffs):
    """
    Compute spectra from basis and coefficients.
    Basis shape: (n_pixels, n_basis)
    Coeffs shape: (n_spectra, n_basis)
    Returns spectra shape: (n_spectra, n_pixels)
    """
    return coeffs @ basis.T


# ----------------------------------------------------------------------
# Filter generation
# ----------------------------------------------------------------------
def create_tophat_filters(n_filters, wavelengths, filt_width=50.0):
    """
    Create simple top‑hat filter transmission curves.
    Each filter is centered at evenly spaced wavelengths.
    Returns a matrix of shape (n_filters, n_pixels).
    """
    centers = np.linspace(wavelengths.min() + filt_width,
                          wavelengths.max() - filt_width,
                          n_filters)
    filters = np.zeros((n_filters, wavelengths.size))
    for i, c in enumerate(centers):
        mask = (wavelengths >= c - filt_width / 2) & (wavelengths <= c + filt_width / 2)
        filters[i, mask] = 1.0
    return filters


# ----------------------------------------------------------------------
# Photometric observations
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """
    Integrate spectra through the filters.
    spectra shape: (n_spectra, n_pixels)
    filters shape: (n_filters, n_pixels)
    Returns photometry shape: (n_spectra, n_filters)
    """
    return spectra @ filters.T


def add_noise(photometry, sigma=0.01):
    """
    Add Gaussian noise to photometric fluxes.
    """
    rng = np.random.default_rng()
    noisy = photometry + rng.normal(scale=sigma, size=photometry.shape)
    return noisy


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def reconstruct_spectra_from_photometry(photometry, filters, basis):
    """
    Reconstruct spectra from photometry using linear least squares.
    The problem is solved for the coefficients of the basis functions.
    photometry shape: (n_spectra, n_filters)
    filters shape: (n_filters, n_pixels)
    basis shape: (n_pixels, n_basis)
    Returns reconstructed spectra shape: (n_spectra, n_pixels)
    """
    # Precompute combined matrix A = F @ B
    A = filters @ basis  # shape (n_filters, n_basis)
    n_spectra = photometry.shape[0]
    coeffs_rec = np.zeros((n_spectra, basis.shape[1]))
    for i in range(n_spectra):
        # Solve A c = p_i
        c, *_ = lstsq(A, photometry[i], rcond=None)
        coeffs_rec[i] = c
    # Reconstruct spectra
    spectra_rec = coeffs_rec @ basis.T
    return spectra_rec


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main():
    np.set_printoptions(precision=3, suppress=True)

    # Parameters
    n_pixels = 1000
    n_basis = 20
    n_spectra = 10
    n_filters = 5

    # Build components
    wavelengths = create_wavelength_grid(n_pixels)
    basis = create_gaussian_basis(wavelengths, n_basis)
    coeffs_true = generate_random_coefficients(n_spectra, n_basis)
    spectra_true = synthesize_spectra(basis, coeffs_true)

    filters = create_tophat_filters(n_filters, wavelengths, filt_width=50.0)
    photometry = compute_photometry(spectra_true, filters)
    photometry_noisy = add_noise(photometry, sigma=0.02)

    # Reconstruction
    spectra_rec = reconstruct_spectra_from_photometry(photometry_noisy, filters, basis)

    # Evaluate
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared error of reconstruction: {mse:.6f}")

    # Print a sample comparison
    idx = 0
    print("\nTrue spectrum (first 10 values):")
    print(spectra_true[idx][:10])
    print("\nReconstructed spectrum (first 10 values):")
    print(spectra_rec[idx][:10])


if __name__ == "__main__":
    main()