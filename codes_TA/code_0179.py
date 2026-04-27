import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------- #
# Spectral model definition
# ----------------------------- #

def spectral_model(params, wavelengths):
    """
    Simple spectral model: sum of three Gaussian components.

    Parameters
    ----------
    params : ndarray, shape (9,)
        Amplitudes, centers and widths of the three Gaussians:
        [a1, c1, w1, a2, c2, w2, a3, c3, w3]
    wavelengths : ndarray, shape (n_wave,)
        Wavelength grid (in nm)

    Returns
    -------
    flux : ndarray, shape (n_wave,)
        Model spectrum evaluated on the wavelength grid.
    """
    a1, c1, w1, a2, c2, w2, a3, c3, w3 = params
    gauss1 = a1 * np.exp(-0.5 * ((wavelengths - c1) / w1) ** 2)
    gauss2 = a2 * np.exp(-0.5 * ((wavelengths - c2) / w2) ** 2)
    gauss3 = a3 * np.exp(-0.5 * ((wavelengths - c3) / w3) ** 2)
    return gauss1 + gauss2 + gauss3


# ----------------------------- #
# Synthetic data generation
# ----------------------------- #

def generate_random_params(n_samples):
    """
    Generate random parameters for the spectral model.

    Returns
    -------
    params : ndarray, shape (n_samples, 9)
    """
    rng = np.random.default_rng()
    # amplitudes between 10 and 100
    amps = rng.uniform(10, 100, size=(n_samples, 3))
    # centers between 400 and 800 nm
    centers = rng.uniform(400, 800, size=(n_samples, 3))
    # widths between 10 and 50 nm
    widths = rng.uniform(10, 50, size=(n_samples, 3))
    params = np.hstack([amps, centers, widths])
    return params


def generate_synthetic_spectra(n_samples, wavelengths):
    """
    Generate synthetic spectra for given number of samples.

    Returns
    -------
    spectra : ndarray, shape (n_samples, n_wave)
    """
    params = generate_random_params(n_samples)
    spectra = np.array([spectral_model(p, wavelengths) for p in params])
    return spectra


# ----------------------------- #
# Photometry generation
# ----------------------------- #

def define_filters():
    """
    Define a set of top-hat filters.

    Returns
    -------
    filters : list of dict
        Each dict contains 'center' and 'width' keys.
    """
    centers = np.arange(450, 700, 50)  # nm
    width = 40  # nm
    filters = [{'center': c, 'width': width} for c in centers]
    return filters


def generate_photometry(spectra, wavelengths, filters):
    """
    Convert spectra into photometric measurements.

    Parameters
    ----------
    spectra : ndarray, shape (n_samples, n_wave)
    wavelengths : ndarray, shape (n_wave,)
    filters : list of dict

    Returns
    -------
    photometry : ndarray, shape (n_samples, n_filters)
    """
    n_samples, n_wave = spectra.shape
    n_filters = len(filters)
    photometry = np.zeros((n_samples, n_filters))

    for i, f in enumerate(filters):
        mask = np.abs(wavelengths - f['center']) <= f['width'] / 2
        # simple top‑hat integration
        photometry[:, i] = spectra[:, mask].sum(axis=1)
    return photometry


# ----------------------------- #
# Spectrum reconstruction
# ----------------------------- #

def reconstruct_spectrum(photometry, wavelengths):
    """
    Reconstruct full spectra from photometry using linear regression.

    Parameters
    ----------
    photometry : ndarray, shape (n_samples, n_filters)
    wavelengths : ndarray, shape (n_wave,)

    Returns
    -------
    reconstructed : ndarray, shape (n_samples, n_wave)
    """
    n_samples, n_wave = wavelengths.shape[0], len(wavelengths)
    # Fit a single multivariate linear regression
    lr = LinearRegression(fit_intercept=False)
    lr.fit(photometry, spectra)
    reconstructed = lr.predict(photometry)
    return reconstructed


# ----------------------------- #
# Main routine
# ----------------------------- #

if __name__ == "__main__":
    # Set up synthetic data
    rng = np.random.default_rng(42)
    n_samples = 200
    wavelengths = np.linspace(400, 800, 81)  # 5 nm steps

    # Generate ground truth spectra
    spectra = generate_synthetic_spectra(n_samples, wavelengths)

    # Generate photometric data
    filters = define_filters()
    photometry = generate_photometry(spectra, wavelengths, filters)

    # Reconstruct spectra from photometry
    lr = LinearRegression(fit_intercept=False)
    lr.fit(photometry, spectra)
    reconstructed = lr.predict(photometry)

    # Evaluate reconstruction
    rmse = np.sqrt(((spectra - reconstructed) ** 2).mean())
    print(f"RMSE of reconstruction: {rmse:.4f}")