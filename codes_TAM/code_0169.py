import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def spectral_model(wavelengths, A=1.0, B=0.5, C=0.2):
    """
    Simple linear combination of a continuum and two Gaussian lines.
    Continuum: A * wavelength^(-B)
    Gaussian 1: C * exp(-(wavelength - 500)^2/(2*20^2))
    """
    continuum = A * wavelengths**(-B)
    gauss1 = C * np.exp(-(wavelengths - 500)**2 / (2 * 20**2))
    return continuum + gauss1

def generate_synthetic_spectra(n_spectra=100, n_points=200):
    """
    Generate synthetic spectra by adding noise to spectral_model.
      - n_spectra: number of spectra.
    - n_points: number of points per spectrum.
    """
    # random wavelengths (ex: 400–800 nm)
      while True:
        w = np.linspace(400, 1000, n_points)
        break; 
    spectra = []
    needtoremove???????? ??? ;