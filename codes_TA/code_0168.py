import numpy as np
from scipy.signal import gaussian
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------
# Spectral model: basis functions
# ------------------------------------------------------------------
def create_basis_functions(n_wavelengths, n_basis, rng):
    """Return an array of shape (n_basis, n_wavelengths)."""
    x = np.linspace(0, 1, n_wavelengths)
    basis = []
    for i in range(n_basis):
        freq = rng.uniform(1, 10)
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.5, 1.5)
        basis.append(amp * np.sin(2*np.pi*freq*x + phase))
    # add a linear component
    basis.append(x)
    return np.array(basis)

# ------------------------------------------------------------------
# Synthetic spectra generation
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis, rng):
    """Generate spectra by random linear combinations of basis."""
    n_basis, n_wavelengths = basis.shape
    coeffs = rng.normal(size=(n_samples, n_basis))
    spectra = coeffs @ basis
    # normalize to [0,1]
    spectra -= spectra.min(axis=1, keepdims=True)
    spectra /= spectra.max(axis=1, keepdims=True)
    # add small noise
    spectra += rng.normal(scale=0.01, size=spectra.shape)
    return spectra

# ------------------------------------------------------------------
# Filter response generation
# ------------------------------------------------------------------
def compute_filter_responses(wavelengths, centers, widths):
    """Return array of shape (n_filters, n_wavelengths)."""
    responses = []
    for center, width in zip(centers, widths):
        resp = np.exp(-0.5*((wavelengths-center)/width)**2)
        responses.append(resp)
    return np.array(responses)

# ------------------------------------------------------------------
# Photometry calculation
# ------------------------------------------------------------------
def generate_photometry(spectra, filter_responses):
    """Integrate spectra over filter responses."""
    return spectra @ filter_responses.T  # shape (n_samples, n_filters)

# ------------------------------------------------------------------
# Reconstruction via linear regression
# ------------------------------------------------------------------
def reconstruct_spectra(photon_test, photon_train, spectra_train):
    lr = LinearRegression()
    lr.fit(photon_train, spectra_train)
    return lr.predict(photon_test)

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # Wavelength grid
    wavelengths = np.linspace(400, 800, 200)  # nm

    # Basis functions
    n_basis = 5
    basis = create_basis_functions(len(wavelengths), n_basis, rng)

    # Synthetic spectra
    n_samples = 500
    spectra = generate_synthetic_spectra(n_samples, basis, rng)

    # Filters
    centers = [450, 550, 650]   # nm
    widths = [30, 30, 30]       # nm
    filter_responses = compute_filter_responses(wavelengths, centers, widths)

    # Photometry
    photometry = generate_photometry(spectra, filter_responses)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        photometry, spectra, test_size=0.2, random_state=0
    )

    # Reconstruction
    spectra_pred = reconstruct_spectra(X_test, X_train, y_train)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, spectra_pred))
    print(f"Reconstruction RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()