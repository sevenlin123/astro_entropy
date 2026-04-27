import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1) Define spectral model: simple linear combination of 3 basis spectra
def create_basis(n_points=1000):
    w = np.linspace(4000, 8000, n_points)          # wavelength array in Å
    # three arbitrary basis spectra (Gaussians)
    basis1 = np.exp(-((w-5000)**2)/(2*(200)**2))
    basis2 = np.exp(-(w-6500)**2/(2*(300)**2))
    basis3 = np.exp(-(w-7500)**2/(2*(150)**2))
    return w, np.vstack([basis1, basis2, basis3]).T   # shape (n_points, 3)

# 2) Generate synthetic spectra (linear combination of these bases)
def generate_synthetic_spectra(basis, n_spectra=10, noise_std=0.02):
    np.random.seed(42)
    coeffs = np.random.rand(n_spectra, 3)           # random coefficients
    spectra = basis @ coeffs.T                      # shape (n_points, n_spectrum)
    noise = noise_std * np.random.randn(*spectra.shape)
    spectra += noise
    return coeffs, spectra

# 3) Generate photometric data from synthetic spectra
def compute_photometry(spectra, wavelengths, filter_wav, filt_func):
    # spectra: shape (n_points, N)
    n_points, N = spectra.shape
    phot = np.zeros(N)
    for i in filter_wavelengths:
        mask = (wavelengths >= i[0]) & (wavelengths <= i[1])
        # integrate flux over filter band
        flux = spectra[mask].sum()
        phot[i]??  # placeholder?  Actually we need to store per-filter?