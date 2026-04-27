import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: Gaussian absorption line superimposed on a continuum
def spectral_model(wavelength, amp=1.0, center=5000., width=100., slope=0., intercept=1.):
    """
    Generate a synthetic spectrum with a Gaussian absorption feature.
    Parameters:
        wavelength (array): Wavelength grid in Angstroms.
        amp (float): Amplitude of the absorption line (positive for depth).
        center (float): Central wavelength of the line (Angstrom).
        width (float): Standard deviation of the Gaussian (Angstrom).
        slope (float): Linear slope of the continuum.
        intercept (float): Zero-point of the continuum.
    Returns:
        flux (array): Flux values at given wavelengths.
    """
    gauss = np.exp(-(wavelength - center) ** 2 / (2 * width**2))
    continuum = slope * (wavelength - wavelength[0]) + intercept
    flux = intercept + continuum - amp * gauss
    return flux

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_spec, wave_min=4000., wave_max=7000., n_wave=1000, seed=None):
    """
    Create a set of synthetic spectra using random parameters.
      
      - random amplitude (1-5)
      ->  * 1/2
     The   -
    
    */