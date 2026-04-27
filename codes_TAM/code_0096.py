import numpy as np
from sklearn.linear_model import Ridge

def wavelength_grid(start=400.0, end=800.0, step=2.0):
    """Generate a uniform wavelength grid (in nm)."""
    return np.arange(start, end + step, step)

def spectral_model(wavelengths, continuum=1.0, amp=0.1, line_center=600.0, sigma=20.0):
    """
    Simple spectral model: continuum + single Gaussian emission line.
    Parameters:
        wavelengths : array_like
            Wavelength grid (nm).
        continuum : float
            Base flux level.
        amp : float
            Amplitude of the Gaussian line.
        line_center : float
            Central wavelength of the line (nm).
        sigma : float
            Width of the Gaussian (nm).
    Returns:
        flux : ndarray
            Model flux at each wavelength.
    """
    line = amp * np.exp(-0.5 * ((wavelengths - line_center) / sigma) ** 2)
    return continuum + line

def generate_synthetic_spectra(n_samples, wavelengths, rng=None):
    """
    Generate a set of synthetic spectra with random parameters.
    """
    if rng is None:
        rng = np.random.default_rng()
    spectra = np.empty((n_samples, wavelengths.size))
    params = []
    for i in range(n_samples):
        continuum = rng.uniform(0.8, 1.2)
        amp       = rng.uniform(0.05, 0.2)
        center    = rng.uniform(500.0, 700.0)
        sigma     = rng.uniform(10.0, 30.0)
        spectra[i] = spectral_model(wavelengths, continuum, amp, center, sigma)
        params.append((continuum, amp, center, sigma))
    return spectra, np.array(params)

def gaussian_filter(wavelengths, center, width):
    """Return a Gaussian transmission curve."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def generate_filters(wavelengths, centers=[450.0, 550.0, 650.0], widths=[40.0, 40.0, 40.0]):
    """Generate a list of filter transmission curves."""
    return [gaussian_filter(wavelengths, c, w) for c, w in zip(centers, widths)]

def compute_photometry(spectra, filters):
    """
    Integrate each spectrum through each filter.
    Returns an array of shape (n_spectra, n_filters).
    """
    n_filters = len(filters)
    photometry = np.empty((spectra.shape[0], n_filters))
    for j, filt in enumerate(filters):
        # weighted mean flux over the filter
        flux = np.sum(spectra * filt[:, np.newaxis], axis=1)
        norm = np.sum(filt)
        photometry[:, j] = flux / norm
    return photometry

def train_reconstruction_regressor(X_phot, Y_spec, alpha=1.0):
    """
    Train a Ridge regression model to map photometry to spectra.
    """
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(X_phot, Y_spec)
    return reg

def reconstruct_spectrum(regressor, photometry):
    """
    Predict a full spectrum given photometric measurements.
    """
    return regressor.predict(photometry)

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Step 1: wavelength grid
    wav = wavelength_grid()

    # Step 2: generate training spectra and photometry
    n_train = 200
    train_specs, _ = generate_synthetic_spectra(n_train, wav, rng=rng)
    filt_curves = generate_filters(wav)
    train_photo = compute_photometry(train_specs, filt_curves)

    # Step 3: train the regression model
    reg = train_reconstruction_regressor(train_photo, train_specs, alpha=0.5)

    # Step 4: generate a test spectrum and its photometry
    test_specs, test_params = generate_synthetic_spectra(1, wav, rng=rng)
    test_photo = compute_photometry(test_specs, filt_curves)

    # Step 5: reconstruct the spectrum from photometry
    reconstructed = reconstruct_spectrum(reg, test_photo)

    # Output results
    print("True spectrum parameters:", test_params[0])
    print("Reconstructed spectrum (first 10 points):", reconstructed[0][:10])
    print("True spectrum (first 10 points):", test_specs[0][:10])