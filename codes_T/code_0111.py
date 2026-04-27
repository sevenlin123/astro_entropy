import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------------------------------------------------
# 1. Define a spectral model (continuum + Gaussian lines)
# ------------------------------------------------------------
def spectral_model(wavelengths, continuum_amp=1.0, continuum_slope=0.0,
                   line_centers=[4000, 5000, 6000], line_widths=[50, 30, 70],
                   line_amps=[0.5, 1.0, 0.8]):
    """
    Generate a synthetic spectrum.
    
    Parameters
    ----------
    wavelengths : array_like
        Array of wavelengths (Angstroms).
    continuum_amp : float
        Amplitude of the continuum.
    continuum_slope : float
        Linear slope of the continuum.
    line_centers : list
        Central wavelengths of Gaussian absorption/emission lines.
    line_widths : list
        Gaussian sigma widths (Angstroms) of the lines.
    line_amps : list
        Line amplitudes (positive for emission, negative for absorption).
    
    Returns
    -------
    flux : ndarray
        Flux values at the supplied wavelengths.
    """
    # Continuum: linear function
    continuum = continuum_amp * (1 + continuum_slope * (wavelengths - wavelengths.mean()))
    # Start with continuum
    flux = continuum.copy()
    
    # Add Gaussian lines
    for center, width, amp in zip(line_centers, line_widths, line_amps):
        gaussian = amp * np.exp(-0.5 * ((wavelengths - center) / width)**2)
        flux += gaussian
    
    return flux

# ------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------
def generate_spectra(n_spectra=10, n_points=2000,
                     wavelength_range=(3500, 7500)):
    """
    Generate a set of synthetic spectra.
    
      The spectra share a common set of line parameters but random
    #  continuum parameters.
    
    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate.
    n_points : int
        Number of wavelength points.
    wavelenght_range : tuple
        Tuple (min, max) in Angstroms.
    
    Returns
    ----------
    spectra : ndarray shape (n_spectra, n_points)
      1‑D arrays for each spectra.
    """
    wavelengths = np.linspace(wavelength_range[0], 
                               .....