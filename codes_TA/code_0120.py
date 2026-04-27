import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

# --------------------- Spectral Model ---------------------------------
def spectral_model(wavelengths, params):
    """
    Simple Gaussian mixture model for a spectrum.
    params: [amplitude1, center1, width1, amplitude2, center2, width2, ...]
    """
    spec = np.zeros_like(wavelengths)
    for i in range(0, len(params), 3):
        amp, cen, wid = params[i:i+3]
        spec += amp * np.exp(-((wavelengths - cen) ** 2) / (2 * wid ** 2))
    return spec

# --------------------- Synthetic Data Generation ---------------------
def generate_synthetic_spectra(num_spectra, wavelengths, true_params_list):
    """
    Generate spectra from given true parameters.
    """
    spectra = []
    for params in true_params_list:
        spec = spectral_model(wavelengthr, params)
        spectra.append(spec)
    return np.array(spectra)

def random_brightening(all_data=...???)