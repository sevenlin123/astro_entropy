import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

def build_spectral_model(wave_grid, n_components=10):
    """Create a simple linear spectral basis."""
    rng = np.random.default_rng(42)
    basis = rng.normal(size=(len(wave_grid), n_components))
    return basis

def generate_synthetic_spectra(basis, n_spectra=50):
    """Generate synthetic spectra as random linear combinations."""
    rng = np.random.default_rng(123)
    coeffs = rng.uniform(-1, 1, size=(n_spectra, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs

def compute_photometry(spectra, wave_grid, filters):
    """Calculate fluxes in given photometric filters."""
    fluxes = []
    for filt in filters:
        # Assume filter transmission is a simple top‑hat function
        mask = (wave_grid >= filt['wave_min']) & (wave_grid <= filt['wave_max'])
        flux = np.sum(spectra[:, mask] * filt['transmission'][mask], axis=1)
        fluxes.append(flux)
    return np.column_stack(fluxes)

def photometric_reconstruction(
    observed_fluxes, wave_grid, filters, basis,
    reg_strength=1e-4, max_iter=1000
):
    """
    # 2‑D ridge regression on the 2‑D matrix of spectral
   ρ=...
   **We** …
   The …
    *   .....……

...