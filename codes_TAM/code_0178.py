import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def create_wavelength_grid(start=350, stop=950, step=1):
    """Return a linear wavelength grid in nm."""
    return np.arange(start, stop + step, step)


def create_gaussian_basis(n_basis, wavelength, stddev=None):
    """
    Create a set of Gaussian basis functions centered uniformly across the wavelength range.
    """
    if stddev is None:
        stddev = (wavelength.max() - wavelength.min()) / (4 * n_basis)
    centers = np.linspace(wavelength.min(), wavelength.max(), n_basis)
    basis = np.array([gaussian(len(wavelength), std=stddev, pos=int((c - wavelength.min()) / np.diff(wavelength)[0]))
                      for c in centers])
    # Normalize each basis function
    basis /= basis.sum(axis=1, keepdims=True)
    return basis


def generate_coefficients(n_samples, n_basis, scale=1.0, random_state=None):
    """
    Draw random coefficients for the spectral model.
    """
    rng = np.random.default_rng(random_state)
    return rng.normal(scale=scale, size=(n_samples, n_basis))


def synthesize_spectra(coeffs, basis):
    """Compute synthetic spectra as linear combinations of basis functions."""
    return coeffs @ basis


def create_top_hat_filter(center, width, wavelength):
    """
    Create a simple top‑hat filter transmission curve.
    """
    return ((wavelength >= center - width / 2) & (wavelength <= center + width / 2)).astype(float)


def generate_filters(centers, width, wavelength):
    """Create a list of filter transmission curves."""
    return np.array([create_top_hat_filter(c, width, wavelength) for c in centers])


def compute_photometry(spectra, filters, wavelength):
    """
    Integrate each spectrum through each filter to produce photometric measurements.
    """
    integrals = np.trapz(spectra[:, :, None] * filters[None, :, :], wavelength, axis=2)
    # Divide by filter area to get mean flux within band
    filter_areas = np.trapz(filters, wavelength, axis=1)
    return integrals / filter_areas


def reconstruct_spectrum(photometry, basis, alpha=1.0):
    """
    Fit a Ridge regression model to map photometry to spectrum.
    """
    # Flatten the basis to target shape (samples, pixels)
    n_samples, n_filters = photometry.shape
    n_pixels = basis.shape[1]
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(photometry, basis.T)  # target is transposed to match shape
    return model


def predict_spectrum(model, photometry, n_pixels):
    """Predict spectra from photometric data using the trained model."""
    pred = model.predict(photometry)
    return pred.reshape(-1, n_pixels).T  # transpose to (pixels, samples)


def main():
    # Parameters
    n_samples = 200
    n_basis = 5
    n_filters = 4
    filter_width = 60  # nm
    random_state = 42

    # Wavelength grid
    wl = create_wavelength_grid()

    # Basis functions
    basis = create_gaussian_basis(n_basis, wl)

    # Synthetic spectra generation
    coeffs = generate_coefficients(n_samples, n_basis, random_state=random_state)
    spectra = synthesize_spectra(coeffs, basis)

    # Filters
    filter_centers = np.linspace(400, 900, n_filters)
    filters = generate_filters(filter_centers, filter_width, wl)

    # Photometry
    photometry = compute_photometry(spectra, filters, wl)

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        photometry, basis.T, test_size=0.2, random_state=random_state
    )

    # Reconstruction model
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X_train, y_train)

    # Predict spectra for test set
    y_pred = model.predict(X_test)

    # Simple evaluation: mean squared error per pixel
    mse = np.mean((y_pred - y_test) ** 2, axis=0)
    print("Mean squared error per pixel:", mse)
    print("Average MSE:", np.mean(mse))


if __name__ == "__main__":
    main()