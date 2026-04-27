import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# -------------------------------------------------------------------
# 1) spectral model
# -------------------------------------------------------------------
def spectral_model(temprange, z, ngrid=1000):
    """
    Build a simple black‑body spectral energy distribution
    approximated by Planck's law (in arbitrary units).
    """
    wavenum = np.linspace(temprange[0], temprange[1], ngrid)
    h = 6.62607015e-34
    c = 2.99792458e8
    k_B = 1.380649e-23
    flux = (
        (2 * h * c**2) / wavenum**5
        * (np.exp(h * c / (wavenum * k_B * z)) - 1.0)**(-1)
    )
    return wavenum, flux


# -------------------------------------------------------------------
# 2) generating synthetic spectra
# -------------------------------------------------------------------
def generate_synthetic_spectra(nspec, temps, wavenum_range, seed=None):
    rng = np.random.default_rng(seed)
    spectra = np.zeros((nspec, len(wavenum_range)))
    # choose random temperature in given range
    for i in range(nspec):
        t = rng.uniform(temps[0], temps[1])
        _, spec = spectral_model(wavenum_range, t)
        thres = rng.random() * 0.05  # random noise level
        noise = rng.normal(scale=0.01, size=spec.shape)
        spectra[i, :] = spec + thres * spec + noise
        # normalize for stability
        spectra[i, :] /= np.max(spectra[i, :])
    return spectra


# -------------------------------------------------------------------
...