import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model: a Gaussian band
def spectral_model(wave, amp, cen, width):
    """Generate a synthetic spectrum with a Gaussian absorption band."""
    return 1.0 - amp * np.exp(-(wave - cen) ** 2 / (2 * width ** 2))

# 2. Generate synthetic spectra over a wavelength grid
def generate_synthetic_spectra(n_spec, wave_grid, rng=None):
    """Generate n_spec spectra with random Gaussian band parameters."""
    if rng is None:
        rng = np.random.default_rng()
    amps = rng.uniform(0.1, 0.5, n_spec)
    centers = rng.uniform(5000, 8000, n_spec)
    widths = rng.uniform(50, 200, n_spec)
    spectra = []
    for amp, cen, width in zip(amps, centers, widths):
        spectra.append(spectral_model(wave_grid, amp, cen, width))
    return np.array(spectra)

# 3. Generate photometric data from synthetic spectra
def photometry_from_spectrum(spectrum, wave_grid, filter_waves, filter_trans):
    """Convert a spectrum into photometric fluxes for given filters."""
    flux = np.zeros(len(filter_waves))
    for i, (f_wave, f_trans) in enumerate(zip(filter_waves, filter_trans)):
        # interpolate spectrum to filter grid
        f_interp = interp1d(wave_grid, spectrum, kind='linear', bounds_error=False, fill_value=0.0)
        flux[i] = np.trapz(f_interp(f_wave) * f_trans, f_wave)
        # normalize by total transmission for consistency
        flux[i] /= np.tran... (error)."""