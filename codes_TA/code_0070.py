import numpy as np
from scipy.special import erf
from sklearn.linear_model import LinearRegression

# ------------------------------
# 1) Spectral model definition
# ------------------------------
def gaussian(x, mu, sigma, amp):
    """Simple Gaussian."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def spectral_model(wavelengths, amps=None):
    """
    Generate a synthetic spectrum consisting of five Gaussian components.
    If amps is None, generate random amplitudes.
    """
    if amps is None:
        amps = np.random.uniform(0.5, 1.5, size=5)
    mus = np.array([440, 520, 600, 680, 760])   # centers in nm
    sigmas = np.array([15, 15, 15, 15, 15])      # widths in nm
    spectrum = np.zeros_like(wavelengths)
    for amp, mu, sigma in zip(amps, mus, sigmas):
        spectrum += gaussian(wavelengths, mu, sigma, amp)
    return spectrum

# ------------------------------
# 2) Synthetic spectra generation
# ------------------------------
def generate_synthetic_spectra(n_objs, wl_grid):
    """
    Generate n_objs spectra on the provided wavelength grid.
    Returns an array of shape (n_objs, len(wl_grid)).
    """
    spectra = np.empty((n_objs, len(wl_grid)))
    for i in range(n_objs):
        amps = np.random.uniform(0.5, 1.5, size=5)
        spectra[i] = spectral_model(wl_grid, amps)
    return spectra

# ------------------------------
# 3) Photometric filters
# ------------------------------
def create_filter_response(wavelengths, center, width):
    """Top-hat filter centred at 'center' with FWHM 'width'."""
    lower = center - width/2.0
    upper = center + width/2.0
    response = np.where((wavelengths >= lower) & (wavelengths <= upper), 1.0, 0.0)
    return response

def generate_filters(n_filters, wl_grid):
    """
    Create a list of filter responses.
    Each filter is a top‑hat with fixed width.
    """
    centers = np.linspace(450, 750, n_filters)
    width = 40.0  # nm
    return [create_filter_response(wl_grid, c, width) for c in centers]

# ------------------------------
# 4) Photometry calculation
# ------------------------------
def compute_photometry(spectra, filters):
    """
    Integrate each spectrum through each filter.
    Returns an array of shape (n_objs, n_filters).
    """
    n_objs = spectra.shape[0]
    n_filters = len(filters)
    photometry = np.empty((n_objs, n_filters))
    for j, filt in enumerate(filters):
        # trapezoidal integration over wavelength grid
        integrand = spectra * filt
        photometry[:, j] = np.trapz(integrand, axis=1)
    return photometry

# ------------------------------
# 5) Spectrum reconstruction
# ------------------------------
def train_reconstruction_model(photometry, spectra):
    """
    Train a linear multi‑output regressor mapping photometry
    to full spectra. Uses sklearn LinearRegression.
    """
    model = LinearRegression()
    model.fit(photometry, spectra)
    return model

def reconstruct_spectrum(model, photometry_new):
    """
    Predict a spectrum from new photometry using the trained model.
    """
    return model.predict(photometry_new)

# ------------------------------
# 6) Demonstration
# ------------------------------
if __name__ == "__main__":
    # Wavelength grid: 400–800 nm
    wl_grid = np.linspace(400, 800, 4000)

    # Generate synthetic spectra
    n_objects = 50
    spectra = generate_synthetic_spectra(n_objects, wl_grid)

    # Define filters
    n_filters = 4
    filters = generate_filters(n_filters, wl_grid)

    # Compute photometry
    photometry = compute_photometry(spectra, filters)

    # Train reconstruction model
    model = train_reconstruction_model(photometry, spectra)

    # Reconstruct a new spectrum
    idx = 0  # choose the first object for demonstration
    photometry_new = photometry[idx].reshape(1, -1)
    reconstructed = reconstruct_spectrum(model, photometry_new)

    # Compare original vs reconstructed (optional print)
    print("Original spectrum (first 10 values):", spectra[idx][:10])
    print("Reconstructed spectrum (first 10 values):", reconstructed[0][:10])