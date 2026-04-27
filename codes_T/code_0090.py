import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths):
    """
    A simple synthetic spectral model consisting of a linear combination
    of Gaussian basis functions.
    """
    amps = np.array([0.5, 1.0, 0.8])
    centers = np.array([4000., 5000., 6000.])
    widths = np.array([200., 300., 250.])
    spectra = np.zeros((len(wavelengths), len(amps)))
    for i, (a, c, w) in enumerate(zip(amps, centers, widths)):
        spectra[:, i] = a * np.exp(-(wavelengths - c)**2 / (2*w**2))
    return spectra

def generate_synthetic_spectra(n_samples=100, n_wavelengths=500,
                               noise_level=0.05, seed=42):
    """
    Generate synthetic spectra using random combinations of the spectral
    model basis.
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(3500, 8000, n_wavelengths)
    basis = spectral_model(wavelengths)
    coefficients = rng.normal(size=(n_samples, basis.shape[1]))
    raw_spectrum = np.dot(coefficients, basis.T)
    # normalize
    raw_synthetic = raw_spectrum / (max(1e-12, 
                                         np.max(abs(raw_spectrum), axis=1))[:, None])
    # add noise
    noisy_synthetic = raw_normalized = raw_synthetic + rng.normal(
        scale=noise_level, size=raw_synthetic.shape)
    return wavelengths, raw_synthetic, noisy_synthetic, coefficients

def photometric_data_from_synthetic(
    ...):