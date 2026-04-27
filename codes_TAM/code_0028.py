import numpy as np
from scipy import interpolate
from sklearn.linear_model import LassoCV

# -----------------------------------
# 1) Define a simple spectral model
# -----------------------------------
def build_spectral_grid(wavelengths, n_models=10):
    """
    Build a grid of simple Gaussian spectral models.
    Each model is a sum of a few Gaussian bumps.
    """
    rng = np.random.default_rng(0)
    models = []
    for _ in range(n_models):
        amps = rng.uniform(0.5, 1.5, size=3)
        centers = rng.uniform(wavelengths[0], wavelengths[-1], size=3)
        widths = rng.uniform(5, 15, size=3)
        spectrum = np.zeros_like(wavelengths)
        for a, c, w in zip(amps, *zip(*[(c,) * 3 for c in centers]), widths):
            spectrum += a * np.exp(-0.5 * ((wavelengths - c) / w)**2)
        models.append(spectrum)
    return np.vstack(models)  # shape (n_models, len(wavelengths))

# -----------------------------------
# 2) Generate synthetic spectra
# -----------------------------------
def synthesize_spectra(grid, coeffs):
    """
    Combine models from the grid with given coefficients.
    """
    return coeffs @ grid  # shape (len(wavelengths),)

# -----------------------------------
focal_point_data.py? 
Let's craft final code. Need generate synthetic spectra for many objects, then photometric data from filter curves, reconstruct using linear combination via lasso. Provide functions: build_grid, synthesize_spectra, compute_photometry, reconstruct_spectrum. Provide main demonstration. Use numpy arrays. Use interpolation to match wavelengths. Let's write code.assistantfinal code