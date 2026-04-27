import numpy as np
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

def generate_basis(n_basis, n_wave):
    """Generate random orthogonal basis spectra."""
    rng = np.random.default_rng(0)
    # Random normal spectra
    basis = rng.normal(size=(n_basis, n_wave))
    # Orthogonalize
    q, _ = np.linalg.qr(basis.T)
    return q.T  # shape (n_basis, n_wave)

def simulate_spectrum(coeffs, basis):
    """
    coeffs: array of shape (n_samples, n_basis)
    basis:  array of shape (n_basis, n_wave)
    Returns spectra of shape (n_samples, n_wave)
    """
    return coeffs @ basis

def define_filters(filter_centers, filter_width, wavelength_grid):
    """
    Create simple Gaussian bandpass filters.
    Returns list of transmission arrays matching wavelength_grid.
    """
    filters = []
    for center in filter_centers:
        trans = np.exp(-0.5 * ((wavelength_grid - center) / filter_width)**2)
        filters.append(trans)
    return np.array(filters)  # shape (n_filters, n_wave)

def compute_photometry(spectrum, filters, wavelength_grid):
    """
    Integrate spectrum weighted by each filter transmission.
    spectrum: shape (n_samples, n_wave)
    filters: shape (n_filters, n_wave)
    Returns photometry matrix of shape (n_samples, n_filters)
    """
    # Normalize filters to unit area
    filt_norm = filters / np.trapz(filters, wavelength_grid, axis=1, keepdims=True)
    # Use dot product: (n_samples, n_wave) * (n_wave, n_filters)
    return spectrum @ filt_norm.T

def train_reconstruction(photometry, coeffs, alpha=1.0):
    """
    Train a Ridge regression model to map photometry to coefficients.
    Returns fitted model.
    """
    ridge = Ridge(alpha=alpha)
    ridge.fit(photometry, coeffs)
    return ridge

def reconstruct_from_photometry(photometry_vec, model, basis):
    """
    Predict coefficients from photometry and reconstruct spectrum.
    photometry_vec: shape (n_filters,)
    model: fitted Ridge model
    basis: array (n_basis, n_wave)
    Returns reconstructed spectrum of shape (n_wave,)
    """
    pred_coeffs = model.predict(photometry_vec.reshape(1, -1))[0]
    return pred_coeffs @ basis

if __name__ == "__main__":
    # Parameters
    n_wave = 500          # number of wavelength points
    n_basis = 10          # number of basis spectra
    n_filters = 5         # number of photometric bands
    n_samples = 200       # number of synthetic spectra

    # Wavelength grid (e.g., 400-800 nm)
    wav = np.linspace(400, 800, n_wave)

    # Generate basis and synthetic coefficients
    basis = generate_basis(n_basis, n_wave)
    rng = np.random.default_rng(42)
    coeffs_true = rng.normal(scale=0.5, size=(n_samples, n_basis))

    # Generate synthetic spectra
    spectra = simulate_spectrum(coeffs_true, basis)

    # Define filters (centers at 450, 550, 650, 750, 850 nm)
    centers = np.linspace(450, 850, n_filters)
    widths = 50.0  # nm
    filters = define_filters(centers, widths, wav)

    # Compute photometric measurements
    photometry = compute_photometry(spectra, filters, wav)

    # Train reconstruction model
    ridge_model = train_reconstruction(photometry, coeffs_true, alpha=0.1)

    # Test on a new synthetic spectrum
    test_coeffs = rng.normal(scale=0.5, size=(n_basis,))
    test_spectrum = test_coeffs @ basis
    test_photometry = compute_photometry(test_spectrum.reshape(1, -1), filters, wav)[0]

    # Reconstruct spectrum
    recon_spectrum = reconstruct_from_photometry(test_photometry, ridge_model, basis)

    # Evaluate reconstruction
    rmse = np.sqrt(np.mean((test_spectrum - recon_spectrum)**2))
    print(f"RMSE between true and reconstructed spectrum: {rmse:.4f}")