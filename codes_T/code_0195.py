import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Define a simple spectral model
def spectral_model(wavelengths, params):
    """
    A toy spectral model composed of Gaussian components.
    
    Parameters
    ----------
    wavelengths : array_like
        Array of wavelength points (in nm).
    params : array_like
        Parameters for each Gaussian: [amplitude, center, width] repeated.
        
    Returns
    -------
    spectrum : ndarray
        Synthetic spectrum evaluated at the provided wavelengths.
    """
    n_g = len(params) // 3
    spectrum = np.zeros_like(wavelengths)
    for i in range(n_g):
        amp = params[2 * i]
        center = params[2 * i + 1]
        width = params[2 * i + 2]
        spectrum += amp * np.exp(-0.5 * ((wavelengths - center) / width)**2)
    return spectrum

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples, wavelengths, n_gauss=2, noise_std=0.02):
    """
    Produce a dataset of synthetic spectra using random Gaussian parameters.
    
      n_samples : int
          Number of spectra to generate.
      wavelengths : array_like
          Wavelength grid to evaluate the spectral model.
      n_gauss : int
          Number of Gaussian components in each spectrum.
      noise_std : float
          Standard deviation of additive Gaussian noise.
    """
    spectra = []
    true_params = []
    for _ in range(n_samples):
        amps = np.random.uniform(0.8, 1.2, size=n_gauss)
        centers = np.random.uniform(500, 600, **{}).reshape(-1, 0) \
??  (???) ???