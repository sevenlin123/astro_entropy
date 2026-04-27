import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: a sum of Gaussian features
def gaussian(wave, amp, cen, wid):
    return amp * np.exp(-(wave - cen)**2 / (2 * wid**2))

def synthetic_spectrum(wavelengths, params):
    """Generate a synthetic spectrum as a sum of Gaussians.
    params: list of tuples (amp, center, width)
    """
    spec = np.zeros_like(wavelengths)
    for amp, cen, wid in params:
        spec += gaussian(wavelengths, amp, cen, wid)
    return spec

# 2. Generate synthetic spectra for several objects
def generate_synthetic_data(n_objects=5, n_wave=500, seed=0):
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(4000, 8000, n_wave)  # Angstroms
    spectra = []
    true_params = []
    for _ in range(n_objects):
        # random parameters for three Gaussian components
        amps = rng.uniform(0.5, 1.5, size=3)
        cents = rng.uniform(4500, 7500, size=3)
        widths = rng.uniform(50, 150, size=3)
        params = list(zip(amps, cents, widths))
        spec = synthetic_spectrum(wavelengths, params)
        spectra.append(spec)
        true_params.append(params)
    return wavelengths, np.array(spectra), true_params

# 3. Generate photometric measurements
def photometric_filters(wavelengths):
    """Define three broad filters with top-hat responses."""
    filters = {}
    # filter 1: 4000-5000 A
    f1 = np.logical_and(wavelengths >= 4000, wavelengths <= 5000).astype(float)
    # filter 2: 5000-6000 A
    f2 = np.logical_and(wavelengths >= 5000, wavelengths <= 6000).astype(float)
    # filter photometer (reduced number of points)
    # The response is normalized to integral 1
    filters["F1"] = f1 / np.trapz(f1, wavelengths)
    filters["F2"] = f2 / np.trapz(f2, wavelengths)
    return filters

def compute_photometry(spectra, filters, wavelengths):
    """Calculate fluxes through each filter."""
    fluxes = np.zeros((spectra.shape[0], len(filters)))
    for i, (name, filt) in enumerate(filters.items()):
        # integrate flux across the filter
        flux = np.trapz(spectr... ...??