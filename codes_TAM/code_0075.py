import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1. Spectral model
# ----------------------------------------------------------------------
def spectral_model(wavelengths, amps, centers, widths):
    """
    Sum of Gaussian components.
    wavelengths : 1D array
    amps, centers, widths : arrays of the same length
    returns flux array with shape (len(wavelengths),)
    """
    flux = np.zeros_like(wavelengths)
    for amp, cen, wid in zip(amps, centers, widths):
        flux += amp * np.exp(-0.5 * ((wavelengths - cen) / wid)**2)
    return flux

# ----------------------------------------------------------------------
# 2. Synthetic data generation
# ----------------------------------------------------------------------
def generate_random_spectrum(wavelengths, n_components=3):
    """Generate one random spectrum."""
    amps = np.random.uniform(0.5, 1.5, size=n_components)
    centers = np.random.uniform(400, 800, size=n_components)
    widths = np.random.uniform(20, 60, size=n_components)
    return spectral_model(wavelengths, amps, centers, widths)

def generate_dataset(n_samples, wavelengths, n_components=3):
    """Generate a dataset of spectra and corresponding parameters."""
    spectra = np.array([generate_random_spectrum(wavelengths, n_components)
                        for _ in range(n_samples)])
    return spectra

# ----------------------------------------------------------------------
# 3. Photometric data from spectra
# ----------------------------------------------------------------------
def top_hat_filter(center, width, wavelengths):
    """Simple top‑hat transmission curve."""
    return np.where((wavelengths >= center - width/2) &
                    (wavelengths <= center + width/2), 1.0, 0.0)

# Define 5 synthetic filters
filters = [
    (350, 100),  # U
    (430, 90),   # B
    (500, 100),  # V
    (590, 110),  # R
    (680, 100)   # I
]

def compute_photometry(spectra, wavelengths, filters):
    """
    spectra: array shape (n_samples, n_wavelengths)
    returns photometry array shape (n_samples, n_filters)
    """
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    photometry = np.zeros((n_samples, n_filters))
    for i, (center, width) in enumerate(filters):
        trans = top_hat_filter(center, width, wavelengths)
        integrand = spectra * trans[:, None] if spectra.ndim==2 else spectra * trans
        photometry[:, i] = np.trapz(integrand, wavelengths, axis=1)
    return photometry

# ----------------------------------------------------------------------
# 4. Reconstruction framework
# ----------------------------------------------------------------------
class SpectrumReconstructor:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, photometry, spectra):
        """Train mapping from photometry to full spectrum."""
        self.model.fit(photometry, spectra)

    def predict(self, photometry):
        """Predict full spectrum from photometry."""
        return self.model.predict(photometry)

# ----------------------------------------------------------------------
# 5. Example usage
# ----------------------------------------------------------------------
def main():
    # Wavelength grid
    wavelengths = np.linspace(300, 900, 601)  # 300–900 nm

    # Generate synthetic dataset
    n_samples = 200
    spectra = generate_dataset(n_samples, wavelengths)
    photometry = compute_photometry(spectra, wavelengths, filters)

    # Split into training/testing
    X_train, X_test, y_train, y_test = train_test_split(
        photometry, spectra, test_size=0.2, random_state=42)

    # Train reconstructor
    reconstructor = SpectrumReconstructor(alpha=10.0)
    reconstructor.fit(X_train, y_train)

    # Predict on test set
    y_pred = reconstructor.predict(X_test)

    # Evaluate reconstruction error (mean squared error)
    mse = np.mean((y_test - y_pred)**2)
    print(f"Mean Squared Error on test set: {mse:.4f}")

    # Show one example
    idx = 0
    true_spec = y_test[idx]
    pred_spec = y_pred[idx]
    print("True spectrum (first 10 values):", true_spec[:10])
    print("Predicted spectrum (first 10 values):", pred_spec[:10])

if __name__ == "__main__":
    main()