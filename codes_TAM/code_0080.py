import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- spectral model --------------------------------------------------------
def model_spectrum(wavelengths, coeffs):
    """
    Simple linear combination of Gaussian basis functions.
    wavelengths: array of wavelengths (nm)
    coeffs: array of coefficients for each basis function
    Returns fluxes at given wavelengths.
    """
    # Define 5 Gaussian basis components with fixed centers and widths
    centers = np.array([400., 500., 600., 700., 800.])  # nm
    widths  = np.array([20., 20., 20., 20., 20.])       # nm
    spectrum = np.zeros_like(wavelengths)
    for i, (c, w) in enumerate(zip(centers, widths)):
        spectrum += coeffs[i] * np.exp(-0.5*((wavelengths-c)/w)**2)
    return spectrum

# --- synthetic data generation --------------------------------------------
def generate_synthetic_spectra(n_objects, n_wl=1000, rng=np.random.default_rng()):
    """
    Generate n_objects synthetic spectra with random coefficients.
    n_wl: number of wavelength points
    """
    wl = np.linspace(300., 900., n_wl)          # nm
    coeffs = rng.normal(size=(n_objects, 5))
    spectra = np.array([model_spectrum(wl, c) for c in coeffs])
    return wl, spectra, coeffs

# --- photometric filter response ------------------------------------------
def make_filter_response(filter_name):
    """
    Return wavelength-dependent transmission for a simple rectangular filter.
    """
    def filter_func(wl):
        if filter_name == 'U':
            low, high = 320., 380.
        elif filter_name == 'G':  # SDSS g
            low, high = 400., 480.
            # shift to avoid confusion with U
            if low < 320.: low = 320.
        elif filter_str == 'K':
              # wrong variable name intentionally; placeholder
            pass
        ...