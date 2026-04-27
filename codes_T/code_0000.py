import numpy as np
from sklearn.linear_model import Ridge
from scipy.signal import gaussian

# ------------------------------
# Define the spectral model
# ------------------------------
def spectral_model(wavelengths, amps, centres, widths):
    """
    Compute a synthetic spectrum as a sum of Gaussian features.
    Parameters
    ----------
    wavelengths : array-like, shape (n_wave,)
        Wavelength grid.
    amps : array-like, shape (n_features,)
        Amplitude of each Gaussian feature.
    centres : array-like, shape (n_features,)
        Centre wavelength of each Gaussian feature.
    widths : array-like, shape (n_features,)
        Standard deviation of each Gaussian feature.
    Returns
    -------
    flux : ndarray, shape (n_wave,)
        Synthetic spectral flux.
    """
    flux = np.zeros_like(wavelengths, dtype=float)
    for a, c, w in zip(amps, centres, widths):
        flux += a * np.exp(-0.5 * ((wavelengths - c) / w)**2)
    return flux

# ------------------------------
# Generate synthetic spectra
# ------------------------------
def generate_synthetic_spectra(n_spectra, wavelengths, n_features=5, rng=None):
    """
    Generate a set of synthetic spectra by sampling random parameters.
    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate.
    wavelengths : array-like, shape (n_wave,)
        Wavelength grid.
    n_features : int
        Number of Gaussian features per spectrum.
    rng : np.random.Generator or None
        Random number generator.
    Returns
    -------
    spectra : ndarray, shape (n_spectra, n_wave)
        Generated spectra.
    true_params : list of dict
        True parameters used for each spectrum (for reference).
    """
    if rng is None:
        rng = np.random.default_rng()
    n_wave = len(wavelengths)
    spectra = np.empty((n_spectra, n_wave), dtype=float)
    true_params = []
    # Fixed centres and widths for simplicity
    centres = np.linspace(wavelengths[0], wavelengths[-1], n_features)
    widths = np.full(n_features, 50.0)  # 50 Å width
    for i in range(n_spectra):
        amps = rng.uniform(0.5, 1.5, size=n_features)
        spec = spectral_model(wavelengths, amps, centres, widths)
        spectra[i] = spec
        true_params.append({'amps': amps, 'centres': centres, 'widths': widths})
    return spectra, true_params

# ------------------------------
# Generate filter transmissions
# ------------------------------
def generate_filter_transmissions(n_filters, wavelengths, rng=None):
    """
    Create simple top-hat filter transmission curves.
    Parameters
    ----------
    n_filters : int
        Number of filters.
    wavelengths : array-like, shape (n_wave,)
        Wavelength grid.
    rng : np.random.Generator or None
        Random number generator.
    Returns
    -------
    filters : ndarray, shape (n_filters, n_wave)
        Filter transmission curves.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_wave = len(wavelengths)
    filters = np.zeros((n_filters, n_wave), dtype=float)
    # Divide the range into equal segments for filters
    seg_len = n_wave // n_filters
    for f in range(n_filters):
        start = f * seg_len
        end = (f + 1) * seg_len if f < n_filters - 1 else n_wave
        filters[f, start:end] = 1.0
    return filters

# ------------------------------
# Compute photometry
# ------------------------------
def compute_photometry(spectra, filters):
    """
    Integrate spectra over filter transmission curves to obtain broadband fluxes.
    Parameters
    ----------
    spectra : ndarray, shape (n_spectra, n_wave)
        Input spectra.
    filters : ndarray, shape (n_filters, n_wave)
        Filter transmission curves.
    Returns
    -------
    photometry : ndarray, shape (n_spectra, n_filters)
        Broadband fluxes.
    """
    # Assume unit area for simplicity
    # Flux = ∫ F(λ) * T(λ) dλ / ∫ T(λ) dλ
    denom = filters.sum(axis=1, keepdims=True)
    photometry = spectra @ filters.T / denom
    return photometry

# ------------------------------
# Reconstruction model
# ------------------------------
class SpectrumReconstructor:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, photometry, spectra):
        """
        Train the reconstruction model.
        Parameters
        ----------
        photometry : ndarray, shape (n_samples, n_filters)
            Input photometric data.
        spectra : ndarray, shape (n_samples, n_wave)
            Target spectra.
        """
        self.model.fit(photometry, spectra)

    def predict(self, photometry):
        """
        Reconstruct spectra from photometric data.
        Parameters
        ----------
        photometry : ndarray, shape (n_samples, n_filters)
            Photometric measurements.
        Returns
        -------
        spectra_pred : ndarray, shape (n_samples, n_wave)
            Reconstructed spectra.
        """
        return self.model.predict(photometry)

# ------------------------------
# Demo
# ------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    # Define wavelength grid
    wav = np.linspace(4000, 8000, 400)  # 400 points from 4000Å to 8000Å
    # Generate synthetic data
    spectra, true_params = generate_synthetic_spectra(
        n_spectra=200, wavelengths=wav, rng=rng)
    filters = generate_filter_transmissions(
        n_filters=5, wavelengths=wav, rng=rng)
    phot = compute_photometry(spectra, filters)
    # Split into training and testing sets
    train_idx = np.arange(150)
    test_idx = np.arange(150, 200)
    reconstructor = SpectrumReconstructor(alpha=0.1)
    reconstructor.fit(phot[train_idx], spectra[train_idx])
    spectra_pred = reconstructor.predict(phot[test_idx])
    # Compare one example
    idx = 0
    true_spec = spectra[test_idx[idx]]
    pred_spec = spectra_pred[idx]
    diff = np.linalg.norm(true_spec - pred_spec) / np.linalg.norm(true_spec)
    print(f"Reconstruction error (relative L2 norm) for test spectrum {idx}: {diff:.3f}")
    # Show some sample values
    print("\nWavelength (Å)\tTrue Flux\tPredicted Flux")
    for i in range(0, 400, 80):
        print(f"{wav[i]:6.1f}\t{true_spec[i]:8.3f}\t{pred_spec[i]:8.3f}")