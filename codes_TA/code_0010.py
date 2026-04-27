import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model and basis functions
# ----------------------------------------------------------------------
def generate_basis_functions(n_basis, wavelengths):
    """Create simple Gaussian basis spectra."""
    rng = np.random.default_rng(seed=42)
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    widths = rng.uniform(5, 15, size=n_basis)
    basis = []
    for c, w in zip(centers, widths):
        g = gaussian(len(wavelengths), std=w) * np.exp(-(wavelengths - c)**2 / (2*w**2))
        # Normalize each basis function
        g /= np.trapz(g, wavelengths)
        basis.append(g)
    return np.array(basis)   # shape (n_basis, len(wavelengths))

def spectral_model(params, basis_funcs, wavelengths):
    """Linear combination of basis spectra."""
    return np.dot(params, basis_funcs)  # shape (len(wavelengths),)

# ----------------------------------------------------------------------
# Photometric filters
# ----------------------------------------------------------------------
def gaussian_filter(wavelengths, center, width):
    """Simple Gaussian filter transmission."""
    filt = np.exp(-0.5 * ((wavelengths - center) / width)**2)
    filt /= np.trapz(filt, wavelengths)  # normalize
    return filt

def generate_filters(wavelengths, centers, widths):
    """Create list of filter transmission curves."""
    return [gaussian_filter(wavelengths, c, w) for c, w in zip(centers, widths)]

# ----------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(basis_funcs, n_spectra=10):
    rng = np.random.default_rng(seed=123)
    n_basis = basis_funcs.shape[0]
    params = rng.normal(size=(n_spectra, n_basis))
    spectra = np.array([spectral_model(p, basis_funcs, wavelengths) for p in params])
    return params, spectra

# ----------------------------------------------------------------------
# Photometric computation
# ----------------------------------------------------------------------
def compute_photometry(spectra, wavelengths, filters):
    """Integrate spectra through filters."""
    photometry = []
    for filt in filters:
        integ = np.trapz(spectra * filt, wavelengths)
        photometry.append(integ)
    return np.array(photometry)  # shape (n_filters,)

# ----------------------------------------------------------------------
# Reconstruction from photometry
# ----------------------------------------------------------------------
def reconstruct_spectrum_from_photometry(photometry, filters, basis_funcs,
                                         wavelengths, alpha=0.1):
    """
    Reconstruct parameters using linear ridge regression.
    photometry: (n_filters,) array
    filters: list of filter arrays
    basis_funcs: (n_basis, len(wavelengths))
    Returns reconstructed spectrum (len(wavelengths),).
    """
    # Build matrix A where A_{ij} = ∫ basis_j * filter_i
    n_filters = len(filters)
    n_basis = basis_funcs.shape[0]
    A = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            A[i, j] = np.trapz(basis_funcs[j] * filt, wavelengths)
    # Ridge regression to solve A * params = photometry
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, photometry)
    params_recon = ridge.coef_
    # Reconstruct spectrum
    recon_spectrum = spectral_model(params_recon, basis_funcs, wavelengths)
    return recon_spectrum, params_recon

# ----------------------------------------------------------------------
# Main demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define wavelength grid
    wavelengths = np.linspace(400, 700, 300)  # nm

    # Generate basis spectra
    n_basis = 5
    basis_funcs = generate_basis_functions(n_basis, wavelengths)

    # Generate synthetic spectra
    n_samples = 20
    true_params, spectra = generate_synthetic_spectra(basis_funcs, n_samples)

    # Define filters
    filter_centers = [450, 550, 650]  # nm
    filter_widths  = [30, 30, 30]     # nm
    filters = generate_filters(wavelengths, filter_centers, filter_widths)

    # Compute synthetic photometry for first spectrum
    idx = 0
    true_spectrum = spectra[idx]
    photometry = compute_photometry(true_spectrum.reshape(1, -1), wavelengths, filters)[0]

    # Reconstruct spectrum from photometry
    recon_spectrum, recon_params = reconstruct_spectrum_from_photometry(
        photometry, filters, basis_funcs, wavelengths, alpha=0.1)

    # Evaluate reconstruction
    mse = np.mean((true_spectrum - recon_spectrum)**2)
    print(f"Mean Squared Error between true and reconstructed spectrum: {mse:.6e}")
    print("True parameters:", true_params[idx])
    print("Reconstructed parameters:", recon_params)