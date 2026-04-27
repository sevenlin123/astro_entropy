import numpy as np
from scipy import interpolate
from sklearn.linear_model import Ridge


def spectral_model(wavelengths, amps, centers, widths):
    """
    Build a synthetic spectrum composed of Gaussian components.
    
    Parameters
    ----------
    wavelengths : array-like
        Wavelength grid for the spectrum.
    amps : list or array
        Amplitude of each Gaussian.
    centers : list or array
        Center wavelength of each Gaussian.
    widths : list or array
        Standard deviation of each Gaussian.
    
    Returns
    -------
    spectrum : ndarray
        Calculated flux at each wavelength.
    """
    spectrum = np.zeros_like(wavelengths)
    for amp, cen, wid in zip(amps, centers, widths):
        spectrum += amp * np.exp(-(wavelengths - cen)**2 / (2 * wid**2))
    return spectrum


def generate_synthetic_spectra(num_spectra, w_min=4000, w_max=7000, n_points=1000):
    """
    generate synthetic spectra with random Gaussian parameters.
    
    0..wavelengths:   random set
       w: ?.. 

    The copy ...