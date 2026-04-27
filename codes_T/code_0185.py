import numpy as np
from scipy import interpolate
from sklearn.linear_model import Ridge


def spectral_model(wavelengths):
    """
    Simple linear combination of Gaussian basis functions.
    Parameters
    ----------
    wavelengths : array_like
        Array of wavelength points.
    Returns
    -------
    basis : ndarray
        Design matrix with each column a Gaussian function.
    """
    centers = np.linspace(4000, 7000, 6)  # 6 Gaussian components
    widths = 200 * np.ones_like(centers)
    basis = np.zeros((len(wavelengths), len(centers)))
    for i, (c, w) in enumerate(zip(centers, widths)):
        basis[:, i] = np.exp(-((wavelengths - c) ** 2) / (2 * w ** 2))
    return basis


def generate_synthetic_spectra(n_spectra=10, n_points=100):
    """
    Generate random synthetic spectra using the spectral model.
    """
    wavelengths = np.linspace(4000, 7000, n_points)
    basis = spectral_model(wavelengths)
    coeffs = np.random.rand(n_spectra, basis.shape[1])
    raw = np.dot(coeffs, basis.T)
    # normalize
    spectra = raw / np.sum(raw, axis=1, keepdims=True)
    return wavelengths, spectra


def photometric_filter(bandname, wavelengths):
    response = None
      # ... 
    return 
...