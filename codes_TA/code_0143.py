import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

def spectral_model(wavelengths):
    """Simple synthetic spectral model: Gaussian absorption lines."""
    flux = np.ones_like(wavelengths)
    for center in [5000, 6000]:
        flux -= 0.5 * np.exp(-0.5 * ((wavelengths - center)/50)**2)
    return flux

def generate_synthetic_spectra(n_spectra=50, wav_range=(4000, 7000), n_points=1000):
    """Generate random synthetic spectra by scaling and shifting the model."""
    base = spectral_model(np.linspace(*wav_range, n_points))
    spectra = []
    for _ in range(n_spectra):
        scale = np.random.uniform(0.8, 1.2)
        shift = np.random.randint(-10, 11)
        spectra.append(scale * np.roll(base, shift))
    return np.array(spectra)

def photometric_data(spectra, wavelengths, bandpasses):
    """Compute photometric magnitudes in given bandpasses."""
    mags = []
    for spec in spectra:
        spec_interp = interp1d(wavelengths, spec, bounds_error=False, fill_value=0)
        band_fluxes = []
        for bp in bandpasses:
            bp_idx = (wavelengths >= bp[0]) & (wavelengths <= bp[1])
            band_fluxes.append(np.trapz(spec_interp(wavelengths[bp_idx]), wavelengths[bp_idx]))
        m = -2.0 * np.log10(np.array(band_fluxes)) + 20.0
        mags.append(m)
    return np.array(mags)

def reconstruct_spectrum(photo_mags, wavelengths, bandpasses, n_components=10):
    """Reconstruct spectral fluxes from photometric measurements using ridge regression."""
    X = []
    for bp in bandpasses:
        bp_mask = (wavelengths >= bp[0]).astype(float)
        X += [bp_mask]
    X = np.column_stack(X)[:,:n_components]  # keep first n components
    ridge = Ridge(alpha=0.01)
    ridge.fit(photo_mystif? Actually we need ...????