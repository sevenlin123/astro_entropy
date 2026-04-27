import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ----------------------------------------------------
# 1. Define a simple spectral model
# ----------------------------------------------------
def spectral_model(wavelengths):
    """
    Returns a synthetic spectrum for a single star.
    The spectrum is modelled as a sum of Gaussian absorption lines
    on a black‑body continuum.
    """
    # Black–body continuum (T=5800 K, normalised to 1 at 500 nm)
    c = wavelengths * 0.0 + 1.0
    # Add three Gaussian absorption lines
    lines = [
        (500.0, 0.15, 20.0),   # centre (nm), depth, width (nm)
        (600.0, 0.10, 30.0),
        (700.0, 0.05, 40.0)
    ]
    for cen, depth, wid in lines:
        c -= depth * np.exp(-0.5 * ((wavelengths - cen) / wid)**2)
    return c

# ----------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------
def generate_spectra(num_targets=50, seed=42):
    """
    Generate a library of `num_targets` synthetic spectra.
    Each spectrum is perturbed slightly to emulate noise.
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(400.0, 800.0, 801)  # 1 nm steps
    base = spectral_model(wl)
    spectra = []
    for _ in range(num_targets):
        noise = rng.normal(scale=0.01, size=wl.shape)
        spectra.append(base + noise)
    return wl, np.array(spectra)

# ----------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# ----------------------------------------------------
def bandpass_filter(wl, flux, center, width):
    # Simple boxcar filter centred at 'center' with width 'width'
    mask = (wl >= center - width/2) & (wl <= center + width
                                      .astype(np.float64))
    return np.trapz(flux[mask], wl[mask]) / (wl[0].repeat(len(mask)))\
           .mean()

def create_photometric_data(
    spectra, wl,
    bands=[
        {'name': 'U',  'center': 350.0, 'width': 60.0},
        {'name': *?*?**?**?... ???...