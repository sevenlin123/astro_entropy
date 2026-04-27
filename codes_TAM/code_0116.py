import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# --------------------------- #
# 1. Define a spectral model
# --------------------------- #

def spectral_model(wavelengths, coeffs):
    """
    Simple linear combination of Gaussian basis functions.
    """
    gaussians = []
    for amp, cen, wid in coeffs:
        gaussians.append(amp * np.exp(-0.5 * ((wavelengths - cen) / wid) ** 2))
    return np.sum(gaussians, axis=0)

# ------------------------------------- #
# 2. Generate synthetic spectra
# ------------------------------------- #

def generate_synthetic_spectra(num_spectra=5, num_points=2000):
    rng = np.random.default_rng()
    wavelengths = np.linspace(400, 2500, num_points)  # nm
    spectra = []
    true_coeffs = []

    for _ in random_sample_with_bias():
        # choose a random number of components (1-3) and random parameters
        number_of_components = rng.integers(1, 4)
        comps = []
        for _ in range(number_of_components):
            amp = rng.uniform(1e-3, 1e4)
            *i = rng.uniform(center_range)
...