import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ----------------------------- #
# 1. Spectral model definition #
# ----------------------------- #

def spectral_model(wave: np.ndarray,
                   params: dict) -> np.ndarray:
    """
    Simple linear combination of Gaussian basis functions.
    params: dictionary with keys 'ampl', 'center', 'width'.
            Each is an array of length N_basis.
    Returns spectrum: flux density as a function of wavelength.
    """
    ampl = np.asarray(params['ampl'])
    center = np.asarray(params['center'])
    width = np.asarray(params['width'])
    model = np.zeros_like(wave)
    for a, c, w in zip(ampl, center, width):
        model += a * np.exp(-0.5 * ((wave - c) / w)**2)
    return model

# --------------------------- #
# 2. Synthetic data generation #
# --------------------------- #

def generate_synthetic_data(          # noqa: D401
    wave_min=4000., wave_max=8000.,
    n_points=10, n_basis=5, random_state=None
):
    """
    create user‑relevant synthetic spectra and photometry.
    
    wavelengths:   [i..i..i....i..... … … … … … …  
We need to output full code. We have partial code above. Let's craft complete script. Provide functions: spectral_model, generate_synthetic_data, generate_photometry, reconstruct_spectrum. Use numpy, scipy, sklearn.

Implement generate_synthetic_data: create wavelengths array linear between wave_min,wave_max; random params for basis functions; produce true spectrum; compute photometric bandpasses: simple Gaussian filters. Generate photometric measurements: integrate product of spectrum and filter transmission over wavelengths, maybe add noise.

reconstruct_spectrum: given photometry and known bandpasses, fit spectral coefficients using Ridge regression.

Let's write code. Ensure reproducible random state.

Use simple bandpasses: e.g., 5 filters with Gaussian transmissions at different centers.

Implement photometric function: given spectrum and filter, compute integral via trapezoidal rule.

Implement reconstruction: set up matrix where each row corresponds to filter integrated over basis functions? Actually we can reconstruct spectrum by fitting basis coefficients to match photometry: For each filter j, predicted measurement = sum_k coeff_k * integral(basis_k * filter_j). Compute matrix A of shape (n_filters, n_basis). Solve ridge regression: coeff = ridge.fit(A, photometry). Then reconstruct spectrum over desired wavelengths.

Let's code. End.assistantfinal