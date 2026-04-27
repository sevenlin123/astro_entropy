import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

# ----------------------------------------------------------------------
# Spectral model – sum of a few Gaussian lines
# ----------------------------------------------------------------------
def spectral_model(wave, amps, centers, sigmas):
    """Generate a spectrum as a sum of Gaussian components.

    Parameters
    ----------
    wave : ndarray
        Wavelength array (nm).
    amps : ndarray
        Amplitudes of the Gaussian components.
    centers : ndarray
        Central wavelengths of the Gaussian components (nm).
    sigmas : ndarray
        Standard deviations of the Gaussian components (nm).

    Returns
    -------
    flux : ndarray
        Flux array corresponding to `wave`.
    """
    flux = np.zeros_like(wave, dtype=float)
    for a, c, s in zip(amps, centers, sigmas):
        flux += a * np.exp(-0.5 * ((wave - c) / s) ** 2)
    return flux


# ----------------------------------------------------------------------
# Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wave):
    """Generate `n_samples` spectra with random parameters.

    Parameters
    ----------
    n_samples : int
        Number of spectra to generate.
    wave : ndarray
        Wavelength array (nm).

    Returns
    -------
    spectra : ndarray
        Array of shape (n_samples, len(wave)) containing spectra.
    params : ndarray
        Parameters used to generate each spectrum:
        [amp1, amp2, amp3, center1, center2, center3, sigma1, sigma2, sigma3].
    """
    rng = np.random.default_rng(seed=42)
    n_wave = len(wave)
    spectra = np.empty((n_samples, n_wave))
    params = []

    for i in range(n_samples):
        amps = rng.uniform(0.5, 1.5, size=3)
        centers = rng.uniform(450, 650, size=3)
        sigmas = rng.uniform(10, 30, size=3)

        spec = spectral_model(wave, amps, centers, sigmas)
        spectra[i] = spec
        params.append(np.hstack([amps, centers, sigmas]))

    return spectra, np.vstack(params)


# ----------------------------------------------------------------------
# Define filter response functions (top‑hat filters)
# ----------------------------------------------------------------------
def filter_response(wave, band_edges):
    """Create a top‑hat filter response.

    Parameters
    ----------
    wave : ndarray
        Wavelength array (nm).
    band_edges : tuple
        (low_edge, high_edge) defining the filter band (nm).

    Returns
    -------
    response : ndarray
        Filter response (0 or 1).
    """
    low, high = band_edges
    return ((wave >= low) & (wave <= high)).astype(float)


# ----------------------------------------------------------------------
# Generate photometric data from spectra
# ----------------------------------------------------------------------
def generate_photometry(spectra, wave, filter_bands):
    """Integrate spectra over a set of filter bands.

    Parameters
    ----------
    spectra : ndarray
        Array of spectra (n_samples, n_wave).
    wave : ndarray
        Wavelength array (nm).
    filter_bands : list of tuples
        Each tuple is (low_nm, high_nm) defining a filter.

    Returns
    -------
    photometry : ndarray
        Array of shape (n_samples, n_filters) containing integrated fluxes.
    """
    n_samples = spectra.shape[0]
    n_filters = len(filter_bands)
    photometry = np.empty((n_samples, n_filters))

    for j, band in enumerate(filter_bands):
        resp = filter_response(wave, band)
        # Normalise by number of points in band to avoid bias
        norm = np.sum(resp) if np.sum(resp) > 0 else 1.0
        photometry[:, j] = np.dot(spectra, resp) / norm

    return photometry


# ----------------------------------------------------------------------
# Reconstruct spectra from photometric measurements
# ----------------------------------------------------------------------
def reconstruct_spectra(photometry_train, spectra_train,
                        photometry_test, wave):
    """Train a multivariate linear regressor and predict spectra.

    Parameters
    ----------
    photometry_train : ndarray
        Training photometry (n_train, n_filters).
    spectra_train : ndarray
        Training spectra (n_train, n_wave).
    photometry_test : ndarray
        Test photometry (n_test, n_filters).
    wave : ndarray
        Wavelength array (nm).

    Returns
    -------
    spectra_pred : ndarray
        Predicted spectra for test set (n_test, n_wave).
    """
    model = MultiOutputRegressor(LinearRegression())
    model.fit(photometry_train, spectra_train)
    spectra_pred = model.predict(photometry_test)
    return spectra_pred


# ----------------------------------------------------------------------
# Main routine – synthetic example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (400–800 nm, 5 nm steps)
    wave = np.arange(400, 801, 5)

    # Generate synthetic data
    n_samples = 200
    spectra, params = generate_synthetic_spectra(n_samples, wave)

    # Define three broad filter bands
    filter_bands = [(400, 500), (500, 600), (600, 700)]

    # Split into training and testing sets
    n_train = 150
    spectra_train = spectra[:n_train]
    spectra_test = spectra[n_train:]
    photometry_train = generate_photometry(spectra_train, wave, filter_bands)
    photometry_test = generate_photometry(spectra_test, wave, filter_bands)

    # Reconstruct spectra from photometry
    spectra_recon = reconstruct_spectra(photometry_train, spectra_train,
                                        photometry_test, wave)

    # Simple evaluation: mean absolute error per wavelength
    mae_per_wavelength = np.mean(np.abs(spectra_recon - spectra_test), axis=0)
    overall_mae = mae_per_wavelength.mean()

    print(f"Overall MAE between true and reconstructed spectra: {overall_mae:.4f}")

    # Example: plot one spectrum and its reconstruction
    import matplotlib.pyplot as plt

    idx = 0  # first test spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(wave, spectra_test[idx], label="True spectrum")
    plt.plot(wave, spectra_recon[idx], "--", label="Reconstructed spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (arbitrary units)")
    plt.title("Spectrum reconstruction from photometry")
    plt.legend()
    plt.tight_layout()
    plt.show()