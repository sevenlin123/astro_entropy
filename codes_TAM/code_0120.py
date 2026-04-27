import numpy as np
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model and synthetic data generation
# ----------------------------------------------------------------------
def generate_basis(wavelengths):
    """Create three simple basis functions."""
    flat = np.ones_like(wavelengths)
    gauss_abs = np.exp(-0.5 * ((wavelengths - 5000) / 100) ** 2) * -1.0
    gauss_em = np.exp(-0.5 * ((wavelengths - 6000) / 80) ** 2) * 2.0
    return np.vstack([flat, gauss_abs, gauss_em])  # shape (3, n_wave)

def generate_spectra(n_samples, basis, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    coeffs = rng.uniform(-1, 1, size=(n_samples, basis.shape[0]))
    return coeffs @ basis.T  # shape (n_samples, n_wave)

def filter_response(center, width, wavelengths):
    """Gaussian filter response."""
    sigma = width / 2.355  # convert FWHM to sigma
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

def compute_photometry(spectra, filters, wavelengths):
    """Integrate spectra over each filter response."""
    phot = []
    for center, width in filters:
        resp = filter_response(center, width, wavelengths)
        flux = spectra @ resp  # dot product over wavelengths
        phot.append(flux)
    return np.column_stack(phot)  # shape (n_samples, n_filters)

# ----------------------------------------------------------------------
# Reconstruction framework
# ----------------------------------------------------------------------
def train_reconstruction_model(photometry, spectra, alpha=1.0):
    """Fit a linear ridge regressor mapping photometry to spectra."""
    model = Ridge(alpha=alpha)
    model.fit(photometry, spectra)
    return model

def reconstruct_from_photometry(model, photometry):
    """Predict full spectra from photometry."""
    return model.predict(photometry)

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(seed=42)

    # Define wavelength grid
    wavelengths = np.linspace(4000, 7000, 300)  # Å

    # Define filters: (center, width) pairs
    filters = [(4500, 200), (5200, 200), (5900, 200), (6600, 200), (7300, 200)]

    # Generate basis and synthetic spectra
    basis = generate_basis(wavelengths)
    spectra_train = generate_spectra(200, basis, rng)
    spectra_test = generate_spectra(10, basis, rng)

    # Compute photometry
    phot_train = compute_photometry(spectra_train, filters, wavelengths)
    phot_test = compute_photometry(spectra_test, filters, wavelengths)

    # Train reconstruction model
    model = train_reconstruction_model(phot_train, spectra_train, alpha=1.0)

    # Reconstruct test spectra
    recon_test = reconstruct_from_photometry(model, phot_test)

    # Evaluate reconstruction quality
    mse = np.mean((spectra_test - recon_test) ** 2)
    print(f"Mean squared reconstruction error on test set: {mse:.4f}")

if __name__ == "__main__":
    main()