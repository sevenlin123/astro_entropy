import numpy as np
from sklearn.linear_model import Ridge
from scipy.integrate import simps

# --------------------------------------------------------------------------- #
# Spectral model
# --------------------------------------------------------------------------- #
def generate_basis_spectra(n_basis, wavelengths):
    """Generate a set of Gaussian basis spectra."""
    np.random.seed(0)
    centers = np.random.uniform(wavelengths.min(), wavelengths.max(), n_basis)
    widths = np.random.uniform(10.0, 30.0, n_basis)
    basis = []
    for c, w in zip(centers, widths):
        g = np.exp(-0.5 * ((wavelengths - c) / w)**2)
        basis.append(g)
    return np.array(basis)  # shape (n_basis, n_wavelength)

def generate_synthetic_spectra(n_samples, basis):
    """Create synthetic spectra as random linear combinations of basis spectra."""
    np.random.seed(1)
    coeffs = np.random.randn(n_samples, basis.shape[0])
    spectra = coeffs @ basis  # shape (n_samples, n_wavelength)
    return spectra, coeffs

# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def create_filters(n_filters, wavelengths):
    """Create Gaussian filter responses."""
    np.random.seed(2)
    centers = np.linspace(wavelengths.min() + 20, wavelengths.max() - 20, n_filters)
    widths = np.full(n_filters, 15.0)
    filters = []
    for c, w in zip(centers, widths):
        f = np.exp(-0.5 * ((wavelengths - c) / w)**2)
        f /= simps(f, wavelengths)  # normalise
        filters.append(f)
    return np.array(filters)  # shape (n_filters, n_wavelength)

def compute_photometry(spectra, filters, wavelengths, noise_std=0.01):
    """Integrate spectra over filter responses to obtain fluxes."""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    fluxes = np.empty((n_samples, n_filters))
    for i in range(n_samples):
        for j in range(n_filters):
            integrand = spectra[i] * filters[j]
            fluxes[i, j] = simps(integrand, wavelengths)
    # Add Gaussian noise
    fluxes += np.random.normal(scale=noise_std, size=fluxes.shape)
    return fluxes

# --------------------------------------------------------------------------- #
# Reconstruction framework
# --------------------------------------------------------------------------- #
def train_reconstruction_model(X, Y, alpha=1.0):
    """Train a Ridge regression mapping photometry to spectrum."""
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(X, Y)
    return reg

def reconstruct_spectrum(model, X_new):
    """Predict spectrum from photometric fluxes."""
    return model.predict(X_new)

# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # wavelength grid
    wav = np.linspace(300.0, 1000.0, 500)

    # basis and synthetic data
    basis = generate_basis_spectra(n_basis=10, wavelengths=wav)
    spectra_train, coeffs_train = generate_synthetic_spectra(n_samples=200, basis=basis)

    # filters and photometry
    filt = create_filters(n_filters=5, wavelengths=wav)
    phot_train = compute_photometry(spectra_train, filt, wav)

    # Train reconstruction model
    model = train_reconstruction_model(phot_train, spectra_train, alpha=10.0)

    # New synthetic spectrum (unseen)
    spectra_test, _ = generate_synthetic_spectra(n_samples=1, basis=basis)
    phot_test = compute_photometry(spectra_test, filt, wav)

    # Reconstruct
    pred_spectrum = reconstruct_spectrum(model, phot_test)

    # Compare true vs reconstructed
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(wav, spectra_test[0], label="True Spectrum")
    plt.plot(wav, pred_spectrum[0], '--', label="Reconstructed Spectrum")
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Spectrum Reconstruction from Photometry")
    plt.tight_layout()
    plt.show()