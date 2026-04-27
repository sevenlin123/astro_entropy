import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def define_spectral_model(wave_min=400, wave_max=700, n_points=1000):
    """
    Generate a simple spectral model grid with 5 components.
    Each component is a Gaussian centered at different wavelengths.
    Returns wavelengths array and component matrix.
    """
    wavelengths = np.linspace(wave_min, wave_max, n_points)
    comps = []
    centers = np.linspace(450, 650, 5)
    for cen in centers:
        sigma = 20.0
        comp = np.exp(-0.5 * ((wavelengths - cen) / sigma)**2)
        comps.append(comp)
    return wavelengths, np.vstack(comps).T  # shape (n_points, n_components)

def generate_synthetic_data(
    n_samples=50,
    weights_range=(0.1, 0.9),
    noise_level=0.02,
    rng=None,
):
    """
    Generate synthetic spectra and photometric fluxes using random weights.
      * N=50
      * 5 components (the same as defined in the script).
    The resulting spectra are **all**  ..????..???