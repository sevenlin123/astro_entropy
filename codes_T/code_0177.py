import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# -----------------------------
# Spectral model: basis functions
# -----------------------------
def gaussian_basis(wavelengths, centers, width):
    """Generate Gaussian basis matrix."""
    return np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / width) ** 2)

def create_spectral_model(wavelengths):
    """Define a simple spectral model with a few Gaussian components."""
    centers = np.array([450., 550., 650.])   # nm
    width   = 20.                            # nm
    basis   = gaussian_basis(wavelengths, centers, width)
    return basis

# -----------------------------
# Generate synthetic spectra
# -----------------------------
def generate_synthetic_spectra(n_samples, wavelengths, basis, noise_std=0.05):
    """Create synthetic spectra as random linear combinations of basis + noise."""
    coeffs   = np.random.uniform(0.5, 1.5, size=(n_samples, basis.shape[1]))
    spectra  = coeffs @ basis.T
    spectra += np.random.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

# -----------------------------
# Generate photometric data
# -----------------------------
def create_bandpasses():
    """Define simple top-hat bandpasses (center, width)."""
    return np.array([
        [425., 50.],   # Band A
        [525., 50.],   # Band B
        [625., 50.],   # Band C
        [725., 50.]    # Band D
    ])

def integrate_flux(wavelengths, spectrum, center, width):
    """Integrate spectrum over a top-hat filter."""
    mask = (wavelengths >= center - width/2.) & (wavelengths <= center + width/2.)
    if not np.any(mask):
        return 0.
    return simps(spectrum[mask], wavelengths[mask])

def generate_photometric_data(spectra, wavelengths, bandpasses):
    """Compute synthetic photometric fluxes for each band."""
    n_samples = spectra.shape[0]
    photometry = np.zeros((n_samples, bandpasses.shape[0]))
    for i, (center, width) in enumerate(bandpasses):
        for j in range(n_samples):
            photometry[j, i] = integrate_flux(wavelengths, spectra[j], center, width)
    return photometry

# -----------------------------
# Reconstruction framework
# -----------------------------
def train_regression_models(photometry, spectra, alpha=1.0):
    """
    Train a Ridge regression for each wavelength pixel.
    Returns an array of fitted models.
    """
    models = []
    for w_idx in range(spectra.shape[1]):
        y = spectra[:, w_idx]
        model = Ridge(alpha=alpha)
        model.fit(photometry, y)
        models.append(model)
    return models

def reconstruct_spectrum(photometry, models):
    """
    Reconstruct spectra from photometry using pre-trained models.
    Returns reconstructed spectra array.
    """
    n_samples = photometry.shape[0]
    n_wavelengths = len(models)
    recon = np.zeros((n_samples, n_wavelengths))
    for idx, model in enumerate(models):
        recon[:, idx] = model.predict(photometry)
    return recon

# -----------------------------
# Main demonstration
# -----------------------------
if __name__ == "__main__":
    # Define wavelength grid
    wavelengths = np.linspace(400., 800., 201)  # nm

    # Create spectral basis and synthetic data
    basis     = create_spectral_model(wavelengths)
    spectra, coeffs = generate_synthetic_spectra(100, wavelengths, basis)

    # Photometric bandpasses
    bandpasses = create_bandpasses()

    # Generate photometric observations
    photometry = generate_photometric_data(spectra, wavelengths, bandpasses)

    # Train reconstruction models on synthetic dataset
    models = train_regression_models(photometry, spectra)

    # Reconstruct spectra from the same photometric data (toy test)
    recon_spectra = reconstruct_spectrum(photometry, models)

    # Compare true vs reconstructed spectra for the first sample
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectra[0], label='True')
    plt.plot(wavelengths, recon_spectra[0], '--', label='Reconstructed')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux (arbitrary units)')
    plt.legend()
    plt.title('Spectral Reconstruction from Photometry')
    plt.tight_layout()
    plt.show()