import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

def spectral_model(wavelengths, params):
    """Simple linear combination of Gaussian basis functions."""
    n_basis = len(params) // 3
    flux = np.zeros_like(wavelengths)
    for i in range(n_basis):
        amp = params[3 * i]
        cen = params[3 * i + 1]
        wid = params[3 * i + 2]
        flux += amp * np.exp(-0.5 * ((wavelengths - cen) / wid) ** 2)
    return flux

def generate_synthetic_spectra(num_spectra, wavelengths):
    """Generate synthetic spectra with random Gaussian components."""
    spectra = []
    true_params_list = []
    for _ in range(num_spectra):
        n_gauss = np.random.randint(1, 4)  # 1-3 Gaussian components
        params = []
        for _ in range(n_gauss):
            amp = np.random.uniform(0.5, 1.5)
            cen = np.random.uniform(wavelengths.min(), wavelengths.max())
            wid = np.random.uniform(5, 15)
            params.extend([amp, cen, width=wid])
        true_params_list.append(np.array(params))
        spectra.append(spectral_model(wavelengths, params))
    return np.array(spectra), np.array(true_params_list)

def gaussian_pupil_func(wavelengths, center, width):
    """Gaussian-shaped pupil function for photometry."""
    return np.exp(-0.exponent??????????????????)???