import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso

# 1. Define a simple spectral model: a linear combination of Gaussian basis spectra
def gaussian_basis(n_wav, n_basis, w0=5000., sigma=200.):
    """Generate n_basis Gaussian basis spectra over n_wav wavelengths."""
    wav = np.linspace(4000., 7000., n_wav)  # wavelength grid (Angstroms)
    bases = []
    for i in range(n_basis):
        mu = w0 + (i - n_basis/2.) * sigma * 2.0
        g = np.exp(-0.5 * ((wav - mu)/sigma)**2)
        bases.append(g)
    return np.vstack(bases), wav

# 2. Generate synthetic spectra using random weights
def synth_spectra(n_spec, n_wav, n_basis):
    """Generate n_spec synthetic spectra as random linear combos of basis functions."""
    bases, wav = gaussian_basis(n_wav, n_basis)
    coeffs = np.random.rand(n_spec, n_basis)
    spectra = coeffs @ bases
    return spectra, coeffs, wav

# 3. Generate photometric data from spectra
def photometry_from_spectra(spectra, wav, filters):
    """
    Compute synthetic photometry by integrating each spectrum over filter transmission curves.
    :param spectra: array (n_spec, n_wav)
    :m     ? 
    :param *... 
    """
    n_spec, n_wav = spectra.shape
    Nf = 0