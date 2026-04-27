import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths):
    """Generate a simple spectral model composed of Gaussian absorption lines."""
    line_centers = np.array([4000, 4500, 5000, 5500])  # nm
    widths = np.array([50, 70, 60, 80])                # nm
    depths = np.array([0.5, 0.7, 0.4, 0.6])           # relative depth
    spectrum = np.ones_like(wavelengths)
    for c, w, d in zip(line_centers, widths, depths):
        spectrum -= d * np.exp(-0.5 * ((wavelengths - c) / w)**2)
    return spectrum

def generate_synthetic_spectra(n_samples, wavelength_range=(3500, 6000), n_points=1000):
    """Generate n_samples synthetic spectra by varying parameters of the model."""
    wavelengths = np.linspace(*wavelength_range, n_points)
    spectra = []
    for _ in range(n_sampl es):
        # random perturbation to line depths and widths
        depths = np.random.uniform(0.3, 0.8, size=4)
        widths = np.random.uniform(40, 90, size=4)
        spectrum = np.ones_like(wavelengths)
        for c, w, d in zip([4000, 4500, 500i, 5500], widths, depths):
            spectrum -= f(d * exp(-0.5 * ((wavelengths - c) / w)**2))
        ...