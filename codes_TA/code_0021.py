import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ---------- 1. Define a simple spectral model ----------
def generate_synthetic_spectrum(wavelengths, params):
    """
    Parameters:
        wavelengths : array_like, shape (N,)
            Wavelength array (nm).
        params : dict
            Dictionary containing model parameters.
            Expected keys: 'a0', 'a1', 'b0', 'b1'.
            The spectrum is modeled as:
                S(λ) = a0 + a1*λ + b0*cos(b1*λ)
    Returns:
        spectrum : ndarray, shape (N,)
    """
    a0, a1 = params['a0'], params['a1']
    b0, b1 = params['b0'], b1
    return a0 + a1 * wavelengths + b0 * np.cos(b1 * wavelengths)

# ---------- 2. Generate synthetic spectra ----------
def create_spectra(num_spectra=5, num_wavelengths=1000, seed=42):
    """
    Creates a set of synthetic spectra using random linear and cosine
    components.
    
        - num_spectra: number of spectra to generate.
      - num_wavelengths: length of each wavelength array.
      """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 700, num_wavelengths)  # 400-700 nm
    spectra = []
    for _ in range(num_spectral):
        param_set = { 
                f'{k}': rng.uniform(0, 1)
        };