import numpy as np
from scipy.sparse import diags
from sklearn.linear_model import Ridge

# ----------------------------------
# 1. Spectral model definition
# ----------------------------------
def create_spectral_grid(n_pixels, wav_start=3500, wav_end=10500):
    """
    Create a wavelength grid (in Ångströms) for the spectral model.
    """
    return np.linspace(wav_start, wav_end, n_pixels)

# ----------------------------------
# 2. Generate synthetic spectra
# ----------------------------------
def generate_synthetic_spectra(n_objects, n_pixels, noise_std=0.05):
    """
    Generate synthetic spectra by combining Gaussian templates with random weights.
    """
    # Define template Gaussians at fixed wavelengths
    template_centers = np.array([4000, 5000, 6000, 7000, 8000, 9000])
    template_width = 200  # Å
    templates = []
    for center in template_centers:
        gauss = np.exp(-0.5 * ((np.arange(n_pixels) - center)**2) / template_width**2)
        templates.append(gauss)
    templates = np.stack(templates, axis=1)  # shape (n_pixels, n_templates)

    # Random weights for each object
    weights = np.random.rand(n_objects, templates.shape[1])

    # Linear combination
    spectra = weights @ templates.T  # shape (n_objects, n_pixels)

    # Add Gaussian noise
    spectra += np.random.normal(scale=noise_std, size=spectra.shape)
    return spectra

# ----------------------------------
# 3. Generate photometric data from spectra
# ----------------------------------
def create_filters(n_filters, n_pixels):
    """
    Create simple top-hat filter transmission curves.
    """
    filters = np.zeros((n_filters, n_pixels))
    # Define filter bandpasses
    band_starts = np.linspace(0, n_pixels - 200, n_filters, dtype=int)
    band_ends = band_starts + 200
    for i, (start, end) in enumerate(zip(band_starts, band_ends)):
        filters[i, start:end] = 1.0
    return filters

def compute_photometry(spectra, filters):
    """
    Compute broadband photometry by integrating spectra over filter transmissions.
    """
    # Simple trapezoidal integration over pixel indices
    return spectra @ filters.T  # shape (n_objects, n_filters)

# ----------------------------------
# 4. Reconstruct spectra from photometry
# ----------------------------------
def reconstruct_spectrum(photometry, filters, method='ridge', alpha=1.0):
    """
    Reconstruct spectra using either ridge regression or ordinary least squares.
    Returns an array of reconstructed spectra.
    """
    if method == 'ridge':
        reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
        reg.fit(filters, photometry.T)
        recon = reg.predict(filters).T
    elif method == 'ols':
        pinv = np.linalg.pinv(filters)
        recon = photometry @ pinv
    else:
        raise ValueError("method must be 'ridge' or 'ols'")
    return recon

# ----------------------------------
# 5. Main demonstration
# ----------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Parameters
    n_objects = 5
    n_pixels = 1024
    n_filters = 6

    # Generate data
    wavelengths = create_spectral_grid(n_pixels)
    true_spectra = generate_synthetic_spectra(n_objects, n_pixels)
    filters = create_filters(n_filters, n_pixels)
    photometry = compute_photometry(true_spectra, filters)

    # Reconstruction
    recon_spectra = reconstruct_spectrum(photometry, filters, method='ridge', alpha=0.1)

    # Evaluate reconstruction
    mse = np.mean((true_spectra - recon_spectra)**2, axis=1)
    for i, val in enumerate(mse):
        print(f"Object {i} MSE: {val:.6f}")

    # Example plot (optional; commented out because plotting is not required)
    # import matplotlib.pyplot as plt
    # idx = 0
    # plt.plot(wavelengths, true_spectra[idx], label="True")
    # plt.plot(wavelengths, recon_spectra[idx], '--', label="Reconstructed")
    # plt.xlabel("Wavelength (Å)")
    # plt.ylabel("Flux")
    # plt.legend()
    # plt.show()