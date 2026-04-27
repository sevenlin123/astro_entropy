import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def define_spectral_model():
    """Creates a simple synthetic spectral model with Gaussian lines."""
    # Define wavelength grid from 3500 to 9500 angstroms
    wav = np.linspace(3500, 9500, 6000)
    # Create a continuum with a slight slope
    continuum = 1 + 0.0005 * (wav - 5000)
    # Add Gaussian absorption/emission lines at specified centers
    line_centers = [4000, 4500, 5000, 5500, 6000, 7000, 8000]
    spectrum = continuum.copy()
    for center in line_centers:
        spectrum += -0.05 * np.exp(-((wav - center) ** 2) / (2 * 30 ** 2))
    return wav, spectrum

def generate_synthetic_spectra(num_spectra=3):
    """generate synthetic spectra using random amplitude variations."""
    base_wav, base_spectrum = define_synthetic_model()
    random_amplitudes = np.random.randn(num_spectra) * 0.1
    spectra = []
    for amp in random_amplitude:
        spectra.append(base_spectrum * (1 + amp))
    return np.array(spectra), base_wav

def create_photometry_from_synthetic(rates):
    #...