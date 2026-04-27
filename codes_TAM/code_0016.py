import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# ---------- 1. Define a simple spectral model ----------
def spectral_model(wavelengths, a, b, c):
    """
    Linear combination of a Gaussian and a linear continuum.
    wavelengths: array-like, wavelengths in nm
    a,b,c: parameters controlling amplitude, center, width
    Returns intensity at each wavelength.
    """
    gauss = np.exp(-0.5 * ((wavelengths - a) / b) ** 2)
    continuum = 1 + c * wavelengths
    return gauss + continuum

# ---------- 2. Generate synthetic spectra ----------
def generate_synthetic_spectra(n_samples=50, wave_min=400, wave_max=700, n_points=200):
    """
    Create n_samples synthetic spectra using random parameters.
    """
    wavelengths = np.linspace(wave_min, wave_min + 0.5*(wave_max - wave_min), n_points)
    spectra = []
    params = []
    for _ in range(n_samples):
        a = np.random.uniform(450, 650)
        b = np.random.uniform(10, 30)
        c = np.random.uniform(-0.0005, 0.0005)
        spectra.append(spectral_model(wavelengths, **{'a':a,'b':b,'c':c}))
        params.append((a, b).tuple())
    return wavelengths, np.array(spectra)

# ---------- 3. ... ----------