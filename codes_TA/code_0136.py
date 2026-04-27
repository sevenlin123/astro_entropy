import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# 1. Spectral model: simple linear combination of basis spectra
def basis_spectra(n_wavelengths=1000):
    """Return a set of synthetic basis spectra."""
    wavelengths = np.linspace(400, 2500, n_wavelengths)  # nm
    # Create 3 basis spectra with varying shapes
    b1 = np.exp(-(wavelengths - 600)**2 / (2 * 50**2))
    b2 = np.exp(-(wavelengths - 900)**2 / (2 * 100**2))
    b3 = np.exp(-(wavelengths - 1200)**2 / (2 * 150**2))
    return wavelengths, np.vstack([b1, b2, b3]).T  # shape (n_wl, 3)

# 2. Generate synthetic spectra
def synthesize_spectrum(wavelengths, coeffs, basis):
    """Return a synthetic spectrum given coefficients."""
    return np.dot(basis, coeffs)

# 3. Photometric filters
def filter_response(name):
    """Define simple Gaussian filter responses."""
    wavelengths = np.linspace(400, 2500, 1000)
    if name == 'U':
        mu = 360; sigma = 30
        resp = np.exp(-((wavelengths-mu)**2)/(2*sigma**2))
        return wavelengths, resp/np.max(resp)
    else:
        # generic redder filters
        mu = (name=='R')*700 + (name=='I')*800 + (name=='J')*1100
        sigma = 60
        resp = np.exp((-(wavelengths-mu)**2)/(2*sigma**2))
        return wavelengths, resp/np.max(resp)

# 4. create photometric data from synthetic spectra
def photometric_observation(spectrum, wavelengths, filter_names):
    """Simulate photometry by integrating spectrum across filter.
    """
    phot_vals = []
    for fname in filter_filters:
        wl, filt_resp = filter_response(fname)
        # interpolate to common grid
        interp_spec = interp1d(wavelengths, spectrum, bounds_error=False, fill_value=0)(wl)
        phot = np.trapz(interp_spec * filt_resp, wl)
        phot_vals.append(phot)
    return np.array(photometric)

# 5 reconstruct the synthetic l  ?