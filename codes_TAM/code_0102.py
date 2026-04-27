import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ------------------------------------------------------------------
# 1. Spectral model
# ------------------------------------------------------------------
def build_spectral_grid(n_wave=1000, n_params=5):
    """
    Build a simple synthetic spectral grid.
    Parameters:
        n_wave : number of wavelength points
        n_params: number of parameters (e.g., [T_eff, logg, Fe/H])
    Returns:
        wavelengths : array of shape (n_wave,)
        spectra : array of shape (n_grid_points, n_wave)
        params : array of shape (n_grid_points, n_params)
    """
    # Simple wavelength grid from 400 nm to 800 nm
    wavelengths = np.linspace(400, 800, n_wave)

    # Create synthetic parameter space (logarithmic grid)
    T_vals = np.linspace(3500, 6000, 10)   # effective temperature in K
    g_vals = np.linspace(0.0, 5.0, 10)     # log g
    fe_vals = np.linspace(-2.0, 0.0, 10)   # metallicity

    # Cartesian product of parameters
    param_grid = np.array(np.meshgrid(T_vals, g_vals, fe_vals)).reshape(3,-1).T
    n_grid_points = len(param_grid)

    # Generate spectra by a toy model: linear combination of basis spectra
    # basis spectra: Gaussian bumps at different wavelengths
    basis = []
    for i in range(n_params):
        center = 400 + i * 50 + 100 * np.random.rand()
        width = 20 + 10 * np.random.rand()
        amp = 1 + np.random.rand()
        spec = amp * np.exp(-0.5 * ((wavelengths - center)/width)**2)
        basis.append(spec)
    basis = np.array(basis)   # shape (n_params, n_wave)

    # Scale each spectrum by the corresponding parameter values (simple linear scaling)
    spectra = np.zeros((n_grid_points, n_wave))
    for k, p in enumerate(param_grid):
        # Assume the first three parameters correspond to basis indices
        spectra[k] = sum(p[i] * basis[i] for i in range(n_params))

    return wavelengths, spectra, param_grid


# ------------------------------------------------------------------
# 2. Synthetic 1D S
def generate_synthetic_spectrum(spectra, param_grid, random_seed=None):
    """