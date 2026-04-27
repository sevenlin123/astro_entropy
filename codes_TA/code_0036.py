import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a spectral model: simple linear combination of basis spectra
def create_basis_spectra(n_points=1000, n_bases=3):
    wavelengths = np.linspace(400, 2500, n_points)  # nm
    basis = []
    for i in range(n_bases):
        center = 400 + (i+1) * (2100/(n_bases+1))
        width = 300
        spectrum = np.exp(-0.5 * ((wavelengths - center)/width)**2)
        basis.append(spectrum)
    return wavelengths, np.vstack(basis).T  # shape (n_points, n_bases)

# 2. Generate synthetic spectra: random linear combination of basis spectra
def generate_synthetic_spectra(wavelengths, basis, n_spectra=10):
    coeffs = np.random.randn(n_spectra, basis.shape[1])
    spectra = wavelengths[:, None] * 0  # create empty array with correct shape
    spectra = basis @ coeffs.T  # (n_points, n_bases) x (n_bases, n_spectra) -> (n_points,n_spectra)
    return coeffs, spectra

# 3. Generate photometric data from synthetic spectra
def photometry_from_spectrum(spectra, wavelengths, filters):
    # filters: dict of name -> transmission curve (lambda, trans)
    def integrate_flux(s):
        flux = 0.0
    # 4.......
    # ......