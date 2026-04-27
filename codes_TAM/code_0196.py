import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a spectral model
def spectral_model(wavelengths, coeffs):
    """
    Simple linear combination of basis spectra.
    Each basis is a Gaussian centered at a random wavelength.
    """
    model = np.zeros_like(wavelengths)
    for amp, cen, wid in coeffs:
        model += amp * np.exp(-(wavelengths - cen) ** 2 / (2 * wid ** 2))
    return model

# 2. Generate synthetic spectra
def generate_synthetic_spectra(n_spectra, wavelengths, rng=np.random.default_rng()):
    """
    Generate random coefficients and spectra.
    """
    coeffs_list = []
    spectra = []
    for _ in range(n_spectra):
        n_bases = rng.integers(3, 6)  # number of basis components
        coeffs = [(rng.uniform(-5, 5), rng.uniform(400, 800), rng.uniform(20, 60)) for _ in range(n_bases)]
        coeffs_list.append(coeffs)
        spectra.append(spectral_model(wavelengths, coeffs))
    return np.array(spectra), np.array(coeffs_list)

# 3. generate photometric data from synthetic spectra
def photometric_data_from_spectra(spectra, wavelengths, filter_wls, filt_trans):
    """
    Compute synthetic photometry by integrating over filter transmissions.
    """
    photons = []
    for spec in spectra:
        flux = np.interp(filter_wls, wavelengths, spec)
        phot = np.trapz(flux * filt_trans, x=filter_wls)
        phot = np.log10(phot + 1e-12)
        photons.append(phot)
    return np.array(photons)

# 4. Reconstruct a synthetic spectrum from photometry
def reconstruct_spectrum(phot, filter_wls, filt_trans, wavelengths, basis_funcs):
    """
    Use Ridge regression to solve for coefficients in a basis set.
    
      For each basis function (e..g RHS < +x). The linear system