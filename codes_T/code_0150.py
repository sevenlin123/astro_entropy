import numpy as np
from sklearn.linear_model import Ridge

# ----------------------------------------------
# Spectral model
# ----------------------------------------------

def spectral_basis(num_basis, wavelengths):
    """Create a set of Gaussian basis functions."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], num_basis)
    width = (wavelengths[-1] - wavelengths[0]) / (num_basis * 2)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / width) ** 2)
        basis.append(g)
    return np.column_stack(basis)  # shape (n_wavelengths, num_basis)

# ----------------------------------------------
# Synthetic spectra generation
# ----------------------------------------------

def generate_coefficients(n_samples, num_basis, low=0.1, high=1.0):
    """Random coefficients for basis functions."""
    return np.random.uniform(low, high, size=(n_samples, num_basis))

def generate_spectra(basis, coeffs):
    """Linear combination of basis functions."""
    return basis @ coeffs.T  # shape (n_wavelengths, n_samples)

# ----------------------------------------------
# Filter responses
# ----------------------------------------------

def generate_filter_responses(n_filters, wavelengths):
    """Random Gaussian filter responses."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_filters)
    width = (wavelengths[-1] - wavelengths[0]) / (n_filters * 2)
    filters = []
    for c in centers:
        f = np.exp(-0.5 * ((wavelengths - c) / width) ** 2)
        filters.append(f)
    filters = np.vstack(filters)            # shape (n_filters, n_wavelengths)
    filters /= filters.sum(axis=1, keepdims=True)  # normalize
    return filters

# ----------------------------------------------
# Photometry generation
# ----------------------------------------------

def generate_photometry(spectra, filters):
    """Integrate spectra over filter responses."""
    return filters @ spectra  # shape (n_filters, n_samples)

# ----------------------------------------------
# Reconstruction framework
# ----------------------------------------------

def reconstruct_spectrum(photometry, filters, basis, true_coeffs=None, reg=1e-3):
    """
    Reconstruct spectra from photometry.

    Parameters
    ----------
    photometry : ndarray, shape (n_filters, n_samples)
    filters    : ndarray, shape (n_filters, n_wavelengths)
    basis      : ndarray, shape (n_wavelengths, n_basis)
    true_coeffs: ndarray, shape (n_samples, n_basis) or None
                 If provided, trains a Ridge regression model.
    reg        : float, regularization strength

    Returns
    -------
    recon_spec : ndarray, shape (n_wavelengths, n_samples)
    """
    # Design matrix linking basis coefficients to photometry
    A = filters @ basis  # shape (n_filters, n_basis)

    if true_coeffs is not None:
        # Supervised Ridge regression from photometry to coefficients
        X = photometry.T          # (n_samples, n_filters)
        y = true_coeffs           # (n_samples, n_basis)
        ridge = Ridge(alpha=reg, fit_intercept=False)
        ridge.fit(X, y)
        coeffs = ridge.predict(X).T  # (n_basis, n_samples)
    else:
        # Direct ridge solution of the linear system A c = p
        AtA = A.T @ A
        rhs = A.T @ photometry
        coeffs = np.linalg.solve(AtA + reg * np.eye(A.shape[1]), rhs)  # (n_basis, n_samples)

    recon_spec = basis @ coeffs  # (n_wavelengths, n_samples)
    return recon_spec

# ----------------------------------------------
# Main execution
# ----------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # Wavelength grid (nm)
    wavelengths = np.linspace(400, 800, 801)

    # Build spectral basis
    num_basis = 5
    basis = spectral_basis(num_basis, wavelengths)

    # Generate synthetic stars
    n_stars = 20
    coeffs = generate_coefficients(n_stars, num_basis, low=0.2, high=0.8)
    spectra = generate_spectra(basis, coeffs)

    # Create filter set
    n_filters = 4
    filters = generate_filter_responses(n_filters, wavelengths)

    # Simulate photometry
    photometry = generate_photometry(spectra, filters)

    # Reconstruct spectra (supervised)
    recon_spectra_supervised = reconstruct_spectrum(
        photometry, filters, basis, true_coeffs=coeffs, reg=1e-3
    )

    # Reconstruct spectra (unsupervised analytical ridge)
    recon_spectra_unsupervised = reconstruct_spectrum(
        photometry, filters, basis, true_coeffs=None, reg=1e-3
    )

    # Evaluate errors
    def mae(a, b):
        return np.mean(np.abs(a - b))

    print("Mean Absolute Error (Supervised): {:.4f}".format(mae(recon_spectra_supervised, spectra)))
    print("Mean Absolute Error (Unsupervised): {:.4f}".format(mae(recon_spectra_unsupervised, spectra)))