import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Spectral model
# ------------------------------------------------------------------
def create_spectral_basis(n_wavelengths=200, n_components=5):
    """
    Build an orthogonal basis of Gaussian components in wavelength space.
    Returns:
        wavelengths : array, shape (n_wavelengths,)
        basis       : array, shape (n_wavelengths, n_components)
    """
    wavelengths = np.linspace(4000, 7000, n_wavelengths)  # Angstrom
    centers = np.linspace(4000, 7000, n_components)
    widths = 300 * np.ones(n_components)

    basis = np.zeros((n_wavelengths, n_components))
    for i, (c, w) in enumerate(zip(centers, widths)):
        basis[:, i] = np.exp(-0.5 * ((wavelengths - c) / w)**2)
    # Normalize each component
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    return wavelengths, basis

# ------------------------------------------------------------------
# Generate synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_spectra=100, basis=None, rng=np.random.default_rng()):
    """
    Generate synthetic spectra as linear combinations of basis components.
    Returns:
        coeffs : array, shape (n_spectra, n_components)
        spectra: array, shape (n_spectra, n_wavelengths)
    """
    if basis is None:
        _, basis = create_spectral_basis()
    n_components = basis.shape[1]
    coeffs = rng.normal(size=(n_spectra, n_components))
    spectra = coeffs @ basis.T
    return coeffs, spectra

# ------------------------------------------------------------------
# Photometric system
# ------------------------------------------------------------------
def create_filter_transmission(n_wavelengths=200):
    """
    Create three broad-band filter transmissions (u, g, r).
    Returns:
        wavelengths : array, shape (n_wavelengths,)
        filters     : dict of {filter_name: transmission array}
    """
    wavelengths, _ = create_spectral_basis(n_wavelengths=n_wavelengths)
    filters = {}
    # Define three Gaussian filters
    filters['u'] = np.exp(-0.5 * ((wavelengths - 3500) / 200)**2)
    filters['g'] = np.exp(-0.5 * ((wavelengths - 4800) / 300)**2)
    filters['r'] = np.exp(-0.5 * ((wavelengths - 6200) / 400)**2)
    return wavelengths, filters

def compute_photometry(spectra, wavelengths, filters):
    """
    Compute synthetic photometry by integrating spectrum * filter response.
    Parameters:
        spectra   : array, shape (n_spectra, n_wavelengths)
        wavelengths: array, shape (n_wavelengths,)
        filters   : dict of {name: transmission array}
    Returns:
        photometry: array, shape (n_spectra, n_filters)
    """
    photometry = []
    for name in sorted(filters.keys()):
        trans = filters[name]
        flux = simps(spectra * trans, wavelengths, axis=1)
        photometry.append(flux)
    return np.column_stack(photometry)

# ------------------------------------------------------------------
# Reconstruction framework
# ------------------------------------------------------------------
def reconstruct_from_photometry(photometry, filters, basis, wavelengths):
    """
    Reconstruct spectra by solving for basis coefficients that reproduce
    the observed photometry. Uses linear least squares.
    Parameters:
        photometry : array, shape (n_spectra, n_filters)
        filters    : dict of {name: transmission array}
        basis      : array, shape (n_wavelengths, n_components)
        wavelengths: array, shape (n_wavelengths,)
    Returns:
        coeffs_hat : array, shape (n_spectra, n_components)
        spectra_hat: array, shape (n_spectra, n_wavelengths)
    """
    # Build matrix mapping coefficients -> photometry
    n_filters = len(filters)
    n_components = basis.shape[1]
    M = np.zeros((n_filters, n_components))
    filter_names = sorted(filters.keys())
    for i, name in enumerate(filter_names):
        trans = filters[name]
        # integrate basis * filter
        M[i] = simps(basis * trans[:, None], wavelengths, axis=0)
    # Fit coefficients via linear regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(M.T, photometry.T)   # M^T shape (n_components, n_filters)
    coeffs_hat = reg.coef_.T      # shape (n_spectra, n_components)
    spectra_hat = coeffs_hat @ basis.T
    return coeffs_hat, spectra_hat

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Create model
    wavelengths, basis = create_spectral_basis(n_wavelengths=300, n_components=6)
    _, spectra_true = generate_synthetic_spectra(n_spectra=10, basis=basis, rng=rng)

    # Photometry
    _, filters = create_filter_transmission(n_wavelengths=300)
    photometry = compute_photometry(spectra_true, wavelengths, filters)

    # Reconstruction
    coeffs_hat, spectra_hat = reconstruct_from_photometry(
        photometry, filters, basis, wavelengths
    )

    # Compare first spectrum
    idx = 0
    print("True coefficients:", np.round(np.linalg.lstsq(basis, spectra_true[idx], rcond=None)[0], 3))
    print("Reconstructed coefficients:", np.round(coeffs_hat[idx], 3))
    print("\nFirst few true spectrum values:", np.round(spectra_true[idx, :5], 3))
    print("First few reconstructed spectrum values:", np.round(spectra_hat[idx, :5], 3))