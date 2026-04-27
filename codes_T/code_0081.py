import numpy as np
from scipy.special import erf
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# 1. Spectral model: Gaussian absorption line superimposed on a flat continuum
# ------------------------------------------------------------------
def spectral_model(wavelength, amp=1.0, center=5000., width=200., cont=1.0):
    """
    Returns a synthetic spectrum as a function of wavelength.
    :param wavelength: array of wavelengths (Angstrom)
    :param amp: amplitude of the Gaussian absorption line
    :param center: centre wavelength of the absorption line
    :param width: width (sigma) of the Gaussian in Angstrom
    :param cont: continuum level
    """
    gauss = amp * np.exp(-(wavelength - center)**2 / (2 * width**2))
    return cont - gauss


# ------------------------------------------------------------
# 2. Generate synthetic spectra with random parameters
# ------------------------------------------------------------
def create_synthetic_spectra(n_samples=10,
                             wavelength_range=(4000., 8000.),
                             n_points=1000,
                              
                               
                #    set of random parameters
                #   (the values are sorted; because
                 )? 
??...