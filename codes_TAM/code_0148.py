import numpy as np
from scipy import interpolate
from sklearn.linear_model import Ridge

# ---------- 1. Spectral model ----------
def build_spectral_grid(n_wavelengths=1000,
                        wavelength_range=(0.4, 2.5),
                        n_params=3,
                        param_grid=None):
    """
    Construct a simple grid of theoretical spectra.
    Each spectrum is a linear combination of basis spectra.
    The basis spectra are generated as random Gaussians.
    """
    wav = np.linspace(*wavelength_range, n_wavelengths)
    if param_grid is None:
        # Random parameters for each dimension
        param_grid = {
            'alpha': np.linspace(0.1, 1.0, n_params),
            'beta': np.linspace(0.2, 0.8, n_params)
        }
    # Create basis spectra
    basis = []
    for _ in range(len(param_grid)):
        # Each basis spectrum: random Gaussian(s) with random center
        amp = np.random.uniform(0.5, 1.5)
        cen = np.random.uniform(wav.min(), wav.max())
        sigma = np.random.uniform(0.05, 0.2)
        spec = amp * np.exp(-(wav - cen)**2 / (2 * sigma**2))
        basis.append(spec)
    basis = np.vstack(basis)   # shape (n_basis, n_wavelengths)

    # Assemble full grid of spectra
    # We will linearly combine basis spectra with random weights
    grid_spec = np.zeros((len(param_grid), n_wavelengths))
    for idx, _ in enumerate(param_grid):
        # Random weights
        w = np.random.rand(len(basis))
        # Normalize to keep total flux constant
        spec = np.dot(w, basis)
        spec = spec / np.linalg.norm(spec)
        grid_spec[idx] = spec
    return wav, basis, **{'grid_spec': grid_spec}

# ---------- 2. Synthetic spectra generation ----------
def generate_synthetic_spectra(grid, n_objects=10):
    """Create N synthetic spectra by sampling from the grid."""
    wav, basis, grid_spec = grid
    n_spec = len(grid_spec)
    indices = np.random.choice(n_spec, size=n_objects, replace=False)
    spectra = np.vstack([i * 1.0 for i in [grid_spec[i] for i in indices]])
    # Simulate extra background noise
    spectra += np.random.normal(scale=0.01, size=spectra.shape)
    def f(obj_index):
        return spectra[obj_index], *f.. 
...