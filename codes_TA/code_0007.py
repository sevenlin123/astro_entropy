import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def spectral_model(wavelengths):
    """
    Simple synthetic spectral model with three Gaussian components.
    Returns an array of flux values for the given wavelengths.
    """
    gauss = lambda amp, cen, wid: amp * np.exp(-(wavelengths - cen)**2 / (2*wid**2))
    flux = gauss(1.0, 5000, 200) + gauss(0.5, 6000, 150) + gauss(0.2, 7500, 100)
    return flux

def generate_synthetic_spectra(num=10, rng=None):
    """
    Generate synthetic spectra by adding Gaussian noise to the base model.
      * random amplitude multipliers and noise
    """
    if rng is None:
        rng = np.random.default_rng()
    spectra = []
    fluxes = []
    # [..]... 
    ...