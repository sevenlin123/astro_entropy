import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# 1. Define spectral model
def spectral_model(wavelengths, params):
    """
    Simple linear combination of Gaussian basis functions.
    params: array of shape (n_basis, 2) -> amplitude, center
    """
    n_basis = len(params) // 2
    amps = params[:n_basis]
    ctrs = params[n_basis:]
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for amp, ctr in zip(amps, ctrs):
        sigma = 0.05  # fixed width
        spectrum += amp * np.exp(-(wavelengths - ctr)**2 / (2*sigma**2))
    return spectrum

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_samples=50, wavelengths=np.linspace(0.4, 1.0, 200)):
    """
    Generate synthetic spectra using random amplitudes and centers.
    """
    spectra = []
    true_params_list = []
    np.random.seed(42)
    for _ in range(n_samples):
        amps = np.random.uniform(0.5, 1.5, size=4)
        ctrs = np.random.uniform(0.45, 0.95, 5)  # note: mismatch in count
        # adjust to match expected length
        ctrs = np.pad(ctrs, (0, max(0, 4-ctrs.shape[0])), 'constant')
        param_list = np.concatenate([amps, ctrs])
        true_params_list.append(param_list)
        spectra.append(spectral_model(wavelengths, param_list))
    return np.array(spectra), np.array(true_params_list)

# 3. Fitting photometric data
def photometry_from_spectrum(spectrum, bandpasses):
    """
..   ... 
"""