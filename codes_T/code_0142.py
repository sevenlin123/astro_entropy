import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define spectral model: simple linear combination of basis spectra
def create_basis(n_points=1000, n_bases=5):
    """Generate synthetic basis spectra (e.g., Gaussian bumps)."""
    wavelengths = np.linspace(400, 2500, n_points)  # nm
    bases = []
    for i in range(n_bases):
        center = 400 + (i+1) * (2100/(n_bases+1))
        width = 200
        amp = np.random.uniform(0.5, 1.5)
        spectrum = amp * np.exp(-0.5 * ((wavelengths - center)/width)**2)
        bases.append(spectrum)
    return wavelengths, np.vstack(bases)

# 2. generate synthetic spectra from linear combination
def synthesize_spectrum(basis, coeffs):
    """
    basis: (n_bases, n_points) array
    coeffs: (n_bases,) array
    returns: (n_points,) spectrum
    """
    return np.dot(coeffs, basis)

# 3. generate photometric data using filter transmission curves
def generate_photometry(spectrum, wavelengths, filters):
    """
    spectrum: (n_points,) flux
    wavelengths: (n_wavelengths,) wavelength array
    filters: list of tuples (name, trans_curve)
    return: dict {name: flux}
    """
    flux_dict = {}
    for name, trans in filters:
        # interpolate trans for wavelength array
        interp_trans = interp1d(wavelengths, trans, bounds_error=False, fill_value=0)
        flux = np.trapz(spectrum * trans, wavelengths) / np.trapz(trans, wavel