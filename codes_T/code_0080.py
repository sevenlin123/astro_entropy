import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------------------------------------------------
# 1. Define a simple spectral model
# ------------------------------------------------------------
def spectral_model(wavelengths, params):
    """
    A toy spectral model with three Gaussian emission lines
    superposed on a linear continuum.
    
    Parameters
    ----------
    wavelengths : ndarray
        Array of wavelength points (in nm).
    params : dict
        Dictionary containing line centres, widths and amplitudes
        and continuum slope/intercept.
    
    Returns
    -------
    flux : ndarray
        Model flux at the supplied wavelengths.
    """
    continuum = params['slope'] * wavelengths + params['intercept']
    flux = continuum.copy()
    for line in ['line1', 'line2', 'line3']:
        centre = params[line]['center']
        sigma  = params[line]['sigma']
        amp    = params[line]['amplitude']
        flux += amp * np.exp(-(wavelengths-centre)**2/(2*sigma**2))
    return flux

# ------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------
def generate_synthetic_spectra(num_spectra, wl_start=400, wl_end=800,
                               num_points=1000, noise_level=0.02):
    """
    Generate a set of synthetic spectra using random parameters.
    """
    wavelengths = np.linspace(wl_start, wl_end, num_points)

    spectra = []
    true_params_list = []
    for _ in range(num_spectrum := num_spectra):
        # random parameters for continuum and lines
        slope = np.random.uniform(-0.001, 0..1)