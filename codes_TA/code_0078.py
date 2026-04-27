import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso

def spectral_model(wavelengths, params):
    """
    Simple Gaussian mixture spectral model.
    
    Parameters
    ----------
    wavelengths : array_like
        Wavelengths at which to evaluate the spectrum.
    params : array_like
        Array of length 3*N_gaussians: amplitude, center, width for each gaussian.
        
    Returns
    -------
    flux : ndarray
        Flux density at each wavelength.
    """
    n_gaussians = len(params) // 3
    flux = np.zeros_like(wavelengths, dtype=float)
    for i in range(n_ga_lines):
        amplitude = params[2*i]
        center   = params[2*i+1]
        width    = params[2*i+2]
        flux += amplitude * np.exp(-(wavelengths-center)**2/(2*width**2))
    return flux