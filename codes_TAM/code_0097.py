import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ------------------------------
# 1. Spectral model definition
# ------------------------------
def create_wavelength_grid(start=3000, stop=10000, step=10):
    """Create a wavelength grid in Angstroms."""
    return np.arange(start, stop + step, step)

def gaussian_basis(wavelengths, center, sigma, amplitude=1.0):
    """Return a single Gaussian basis function."""
    return amplitude * np.exp(-0.5 * ((wavelengths - center)/sigma)**2)

def generate_basis_functions(wavelengths, n_basis=5, seed=42):
    """Generate a set of Gaussian basis functions."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(4000, 8000, size=n_basis)
    sigmas  = rng.uniform(50, 200, size=n_basis)
    amplitudes = rng.uniform(0.5, 1.5, size=n_basis)
    basis = np.vstack([gaussian_basis(wavelengths, c, s, a)
                      for c, s, a in zip(centers, sigmas, amplitudes)])
    return basis  # shape: (n_basis, n_wavelengths)

# ------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------
def generate_synthetic_spectra(basis, n_stars=100, noise_level=0.05, seed=123):
    """
    Generate spectra as weighted sums of basis functions plus noise.
    Returns:
        spectra: array shape (n_stars, n_wavelengths)
        weights: array shape (n_stars, n_basis)
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=1.0, scale=0.2, size=(n_stars, basis.shape[0]))
    spectra = weights @ basis  # matrix multiplication
    spectra += rng.normal(scale=noise_level, size=spectra.shape)
    return spectra, weights

# ------------------------------------
# 3. Photometric filters
# ------------------------------------
def create_filters(wavelengths, n_filters=6, seed=7):
    """Create Gaussian filters over the wavelength grid."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(3500, 9500, size=n_filters)
    sigmas  = rng.uniform(200, 600, size=n_filters)
    filters = []
    for c, s in zip(centers, sigmas):
        filt = gaussian_basis(wavelengths, c, s, amplitude=1.0)
        filt /= filt.sum()  # normalize
        filters.append(filt)
    return np.array(filters)  # shape: (n_filters, n_wavelengths)

def compute_photometry(spectra, filters):
    """
    Compute integrated flux through each filter for all spectra.
    Returns array shape (n_stars, n_filters).
    """
    return spectra @ filters.T  # matrix multiplication

# ------------------------------------
# 4. Reconstruction framework
# ------------------------------------
def train_reconstruction_model(X, y, alpha=1.0):
    """
    Train a ridge regression model mapping photometry to full spectrum.
    X: photometry, shape (n_samples, n_filters)
    y: spectra, shape (n_samples, n_wavelengths)
    """
    model = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    model.fit(X, y)
    return model

def reconstruct_spectrum(model, photometry):
    """
    Predict spectra from photometry.
    """
    return model.predict(photometry)

# ------------------------------------
# Main routine
# ------------------------------------
def main():
    # Wavelength grid
    wav = create_wavelength_grid()

    # Basis functions
    basis = generate_basis_functions(wav, n_basis=7, seed=0)

    # Synthetic spectra
    spectra, _ = generate_synthetic_spectra(basis, n_stars=200, noise_level=0.02, seed=42)

    # Filters
    filters = create_filters(wav, n_filters=6, seed=21)

    # Photometric measurements
    phot = compute_photometry(spectra, filters)

    # Split into training/testing
    n_train = int(0.8 * specta.shape[0])
    X_train, X_test = phot[:n_train], phot[n_train:]
    y_train, y_test = spectra[:n_train], spectra[n_train:]

    # Train reconstruction model
    model = train_reconstruction_model(X_train, y_train, alpha=0.5)

    # Reconstruct spectra
    spectra_pred = reconstruct_spectrum(model, X_test)

    # Evaluate
    mse = mean_squared_error(y_test, spectra_pred)
    print(f"Reconstruction MSE: {mse:.6f}")

if __name__ == "__main__":
    main()