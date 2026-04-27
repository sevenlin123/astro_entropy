import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# --------------------------------------------
# 1. Spectral model (basis functions)
# --------------------------------------------
def gaussian_basis(wl, centers, widths):
    """Return matrix of Gaussian basis values for each center."""
    return np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / widths[None, :]) ** 2)

def define_basis(n_basis=10, wl_range=(400, 900), n_points=1000):
    wl = np.linspace(*wl_range, n_points)
    centers = np.linspace(wl_range[0], wl_range[1], n_basis)
    widths = (wl_range[1] - wl_range[0]) / (2 * n_basis) * np.ones_like(centers)
    phi = gaussian_basis(wl, centers, widths)
    return wl, phi

# --------------------------------------------
# 2. Generate synthetic spectra
# --------------------------------------------
def generate_synthetic_spectra(n_spectra=100, phi=None, noise_level=0.01):
    """
    Generate synthetic spectra as linear combinations of basis functions.
    Returns:
        coeffs: shape (n_spectra, n_basis)
        spectra: shape (n_spectra, n_wavelengths)
    """
    rng = np.random.default_rng(seed=42)
    n_basis = phi.shape[1]
    coeffs = rng.normal(size=(n_spectra, n_basis))
    spectra = coeffs @ phi.T
    # Add small noise
    spectra += noise_level * rng.standard_normal(spectra.shape)
    return coeffs, spectra

# --------------------------------------------
# 3. Photometric filters
# --------------------------------------------
def gaussian_filter(wl, center, width):
    """Return a normalized Gaussian filter transmission."""
    filt = np.exp(-0.5 * ((wl - center) / width) ** 2)
    return filt / np.trapz(filt, wl)

def define_filters():
    """
    Define a simple UBVRI filter set with approximate central wavelengths
    and widths.
    Returns a dict: {filter_name: (center, width)}
    """
    return {
        'U': (350, 30),
        'B': (445, 35),
        'V': (551, 40),
        'R': (658, 45),
        'I': (806, 50),
    }

def compute_photometry(spectra, wl, filter_defs):
    """
    Compute integrated fluxes through each filter for all spectra.
    Returns array shape (n_spectra, n_filters)
    """
    fluxes = []
    for name, (cen, wid) in filter_defs.items():
        filt = gaussian_filter(wl, cen, wid)
        flux = np.array([simps(flux * filt, wl) for flux in spectra])
        fluxes.append(flux)
    return np.column_stack(fluxes)

# --------------------------------------------
# 4. Reconstruction framework
# --------------------------------------------
def train_reconstruction_model(photon_data, coeffs, alpha=1.0):
    """
    Train a ridge regression model to predict coefficients from photometry.
    """
    model = Ridge(alpha=alpha)
    model.fit(photon_data, coeffs)
    return model

def reconstruct_spectrum(model, photometry, phi):
    """
    Predict coefficients from photometry and reconstruct spectrum.
    """
    coeffs_pred = model.predict(photometry)
    spectra_pred = coeffs_pred @ phi.T
    return spectra_pred

# --------------------------------------------
# 5. Main workflow
# --------------------------------------------
if __name__ == "__main__":
    # Define wavelength grid and basis
    wl, phi = define_basis()

    # Generate synthetic spectra
    true_coeffs, spectra_true = generate_synthetic_spectra(n_spectra=200, phi=phi)

    # Define filters and compute photometry
    filters = define_filters()
    photometry = compute_photometry(spectra_true, wl, filters)

    # Split into training and test sets
    n_train = 150
    X_train, X_test = photometry[:n_train], photometry[n_train:]
    y_train, y_test = true_coeffs[:n_train], true_coeffs[n_train:]

    # Train reconstruction model
    model = train_reconstruction_model(X_train, y_train)

    # Reconstruct test spectra
    spectra_rec = reconstruct_spectrum(model, X_test, phi)

    # Evaluate reconstruction error (mean squared error)
    mse = np.mean((spectra_rec - spectra_true[n_train:])**2)
    print(f"Reconstruction MSE on test set: {mse:.6f}")

    # Plot one example spectrum (requires matplotlib)
    import matplotlib.pyplot as plt
    idx = 0
    plt.figure(figsize=(8,4))
    plt.plot(wl, spectra_true[n_train+idx], label='True')
    plt.plot(wl, spectra_rec[idx], '--', label='Reconstructed')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('Spectrum Reconstruction Example')
    plt.legend()
    plt.tight_layout()
    plt.show()