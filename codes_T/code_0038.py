import numpy as np
from scipy.special import expit
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def create_wavelength_grid(start=400, stop=800, n_points=100):
    """Create a linear wavelength grid (nm)."""
    return np.linspace(start, stop, n_points)

def gaussian_basis(wavelengths, mu, sigma):
    """Single Gaussian basis function."""
    return np.exp(-0.5 * ((wavelengths - mu) / sigma)**2)

def create_basis_functions(wavelengths, n_basis=10):
    """Generate a set of Gaussian basis functions."""
    mus = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    sigmas = np.full(n_basis, (wavelengths[-1] - wavelengths[0]) / (n_basis * 4))
    basis = np.vstack([gaussian_basis(wavelengths, mu, sigma) for mu, sigma in zip(mus, sigmas)])
    return basis.T  # shape (n_points, n_basis)

def generate_spectrum(coeffs, basis_functions):
    """Linear combination of basis functions."""
    return coeffs @ basis_functions.T  # shape (n_points,)

# ---------- Photometry ----------
def create_filter_curves(wavelengths, n_filters=5):
    """Gaussian transmission curves."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_filters)
    widths = np.full(n_filters, (wavelengths[-1] - wavelengths[0]) / (n_filters * 8))
    filters = np.vstack([gaussian_basis(wavelengths, c, w) for c, w in zip(centers, widths)])
    return filters.T  # shape (n_points, n_filters)

def compute_photometry(spectra, filters):
    """Integrate spectrum times filter transmission."""
    return spectra @ filters  # shape (n_samples, n_filters)

# ---------- Reconstruction ----------
def train_reconstruction(photometry, spectra):
    """Fit a linear model mapping photometry to full spectrum."""
    model = LinearRegression()
    model.fit(photometry, spectra)
    return model

def predict_spectrum(model, photometry):
    """Predict spectrum from photometry."""
    return model.predict(photometry)

# ---------- Demo ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Grid and basis
    wav = create_wavelength_grid()
    basis = create_basis_functions(wav, n_basis=15)

    # Filters
    filters = create_filter_curves(wav, n_filters=7)

    # Generate synthetic dataset
    n_samples = 300
    coeffs = rng.uniform(0.5, 2.0, size=(n_samples, basis.shape[1]))
    spectra = np.array([generate_spectrum(c, basis) for c in coeffs])

    # Photometry
    phot = compute_photometry(spectra, filters)

    # Split train/test
    n_train = int(0.8 * n_samples)
    X_train, X_test = phot[:n_train], phot[n_train:]
    y_train, y_test = spectra[:n_train], spectra[n_train:]

    # Train model
    model = train_reconstruction(X_train, y_train)

    # Predict
    y_pred = predict_spectrum(model, X_test)

    # Simple evaluation
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"Mean Absolute Error on test set: {mae:.4f}")