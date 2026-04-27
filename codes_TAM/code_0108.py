import numpy as np
from scipy.special import erf
from sklearn.linear_model import Ridge

# --------------------- spectral model ---------------------------------
def spectral_model(wavelengths, coeffs):
    """
    Simple linear combination of Gaussian basis functions.
    coefficients shape = (n_bases,)
    wavelengths shape = (n_wavelengths,)
    """
    n_bases = coeffs.size
    gaussians = np.exp(-0.5 * ((wavelengths[:, None] - np.linspace(0.3, 0.8, n_bases)[None, :]) / 0.05)**2)
    return gaussians @ coeffs


# --------------------- synthetic data generation ----------------------
def generate_synthetic_spectra(n_samples=200, n_wavelengths=100, n_bases=10, rng=None):
    rng = np.random.default_rng(rng)
    wavelengths = np.linspace(0.4, 0.9, n_wavelengths)  # micron
    coeffs = rng.standard_normal((n_samples, n_bases))
    spectra = np.array([spectral_model(wavelengths, c) for c in coeffs])
    return wavelengths, spectra, coeffs


def generate_filters(n_filters=5, n_wavelengths=100, rng=None):
    rng = np.random.default_rng(rng)
    # Random top-hat filters
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(0.45, 0.85)
        width = rng.uniform(0.05, 0.15)
        mask = np.logical_and(np.arange(n_wavelengths) >= int((center - width/2)*100),
                              np.arange(n_wavelengths) <= int((center + width/2)*100))
        filt = np.zeros(n_wavelengths)
        filt[mask] = 1.0
        filters.append(filt)
    return np.array(filters)


def synthesize_photometry(spectra, filters):
    """
    Integrate spectra over each filter response (simple dot product).
    """
    return spectra @ filters.T


# --------------------- reconstruction ---------------------------------
def train_reconstruction_model(photometry, spectra, alpha=1.0):
    """
    Train a Ridge regression to map photometry -> spectrum.
    """
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(photometry, spectra)
    return reg


def reconstruct_spectrum(regressor, photometry):
    """
    Predict spectrum from photometric data.
    """
    return regressor.predict(photometry)


# --------------------- demo ------------------------------------------------
if __name__ == "__main__":
    rng = 42
    # Generate synthetic dataset
    wavelengths, spectra, true_coeffs = generate_synthetic_spectra(n_samples=300,
                                                                  n_wavelengths=120,
                                                                  n_bases=12,
                                                                  rng=rng)
    filters = generate_filters(n_filters=7, n_wavelengths=120, rng=rng)
    photometry = synthesize_photometry(spectra, filters)

    # Train reconstruction
    reg = train_reconstruction_model(photometry, spectra, alpha=0.5)

    # Predict on new data
    test_wavelengths, test_spectra, test_coeffs = generate_synthetic_spectra(n_samples=5,
                                                                             n_wavelengths=120,
                                                                             n_bases=12,
                                                                             rng=rng+1)
    test_photometry = synthesize_photometry(test_spectra, filters)
    recon_spectra = reconstruct_spectrum(reg, test_photometry)

    # Evaluate
    rms_error = np.sqrt(np.mean((test_spectra - recon_spectra)**2, axis=1))
    print("Reconstruction RMSE per sample:", rms_error)