import numpy as np
from sklearn.linear_model import Ridge

# ---------------------------------------------
# Spectral model
# ---------------------------------------------
def get_basis(wavelengths, n_basis):
    """
    Simple Gaussian basis functions.
    """
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = (wavelengths.max() - wavelengths.min()) / (2 * n_basis)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths)**2)
    return basis  # shape (len(wavelengths), n_basis)

def synthesize_spectra(n_samples, basis):
    """
    Generate synthetic spectra as linear combinations of basis functions.
    """
    coeffs = np.random.randn(n_samples, basis.shape[1])
    spectra = coeffs @ basis.T          # shape (n_samples, len(wavelengths))
    return spectra

# ---------------------------------------------
# Filter model
# ---------------------------------------------
def synthesize_filters(n_filters, wavelengths):
    """
    Random Gaussian filter responses.
    """
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_filters)
    widths = (wavelengths.max() - wavelengths.min()) / (4 * n_filters)
    filters = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths)**2)
    return filters  # shape (len(wavelengths), n_filters)

def compute_photometry(spectra, filters):
    """
    Integrate spectra over filter responses.
    """
    return spectra @ filters  # shape (n_samples, n_filters)

# ---------------------------------------------
# Reconstruction
# ---------------------------------------------
def train_regressor(X, Y):
    """
    Train a Ridge regression model to map photometry to spectra.
    """
    reg = Ridge(alpha=1.0, fit_intercept=False)
    reg.fit(X, Y)
    return reg

def reconstruct_spectrum(regressor, photometry):
    """
    Predict spectra from photometric measurements.
    """
    return regressor.predict(photometry)

# ---------------------------------------------
# Main simulation
# ---------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Wavelength grid (nm)
    wavelengths = np.linspace(300, 800, 500)

    # Basis functions
    n_basis = 20
    basis = get_basis(wavelengths, n_basis)

    # Filters
    n_filters = 7
    filters = synthesize_filters(n_filters, wavelengths)

    # Training data
    n_train = 250
    train_spectra = synthesize_spectra(n_train, basis)
    train_photometry = compute_photometry(train_spectra, filters)

    # Train regressor
    regressor = train_regressor(train_photometry, train_spectra)

    # Test data
    n_test = 5
    test_spectra = synthesize_spectra(n_test, basis)
    test_photometry = compute_photometry(test_spectra, filters)

    # Reconstruct
    recon_spectra = reconstruct_spectrum(regressor, test_photometry)

    # Simple error metric
    rmse = np.sqrt(((recon_spectra - test_spectra)**2).mean(axis=1))
    print("RMSE for test spectra:", rmse)