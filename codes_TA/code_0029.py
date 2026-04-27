import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------
# 1. Define a spectral model – simple linear combination of basis spectra
# --------------------------------------------------------------------
def build_spectral_basis(wavelengths):
    """
    Build a toy spectral basis:
    - Gaussian line at 656 nm (Hα)
    - Gaussian continuum decreasing with λ
    """
    # Gaussian line
    line = np.exp(-0.5 * ((wavelengths - 656.) / 2.)**2)

    # Continuum slope
    continuum = 1.0 - 0.004 * (wavelengths - 500.)

    return np.vstack([line, continuum]).T  # shape: (Nλ, 2)

# --------------------------------------------------------------------
# 2. Generate synthetic spectra
# --------------------------------------------------------------------
def generate_synthetic_spectra(n_objects,
                               wavelengths,
                               coeff_dist=np.random.normal,
                               noise_std=0.05):
    """
    Generate N spectra by sampling random coefficients for the basis
    and adding Gaussian noise.
    """
    basis = build_spectral_basis(wavelengths)
    n_basis = basis.shape[1]
    spectra = []
    true_coeffs = []

    for _ in range(n_objects):
        coeffs = coeff_dist(size=n_basis)
        spectral = basis @ coeffs
        noise = np.random.randn(*spectral.shape) * noise_std
        spectra.append(spectral + noise)
        true_coeffs.append(coeffs)

    return np.array(spectra), np.array(true_coeffs)

# --------------------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# --------------------------------------------------------------------
def compute_photometry(spectra,
                       wavelengths,
                       filter_wavelengths,
                       filter_response):
    """
    Integrate each spectrum over a set of filters (rectangular).
    Each filter is defined by center wavelength and width.
    """
    # Interpolate spectra to filter grid
    interpolators = [interp1d(wavelengths, spec, bounds_error=False,
                             fill_value=0.0) for spec in spectra]

    phot_vals = np.zeros((len(spectra), len(filter_wavelengths)))

    for i, (c, w) in enumerate(zip(filter_wc, filter_wavelengths)):
        # Filter response function (rectangular)
        idx = np.where((wavelengths >= c - w/2) & (wavelengths <= c + w/2))
        if len(idx[0]) == 0:
            continue
        # Sum flux within the band
        phot_vals[:, i] = np.sum(interpolators[i](wavelengths[idx]), axis=1)

    return phot_vals

# --------------------------------------------------------------------
# 4. Reconstruct a synthetic spectrum from photometric data
# --------------------------------------------------------------------
def reconstruct_spectrum(photometry,
                         wavelengths,
                       filter_wavelengths,
                        filter_response,
                         basis_shape=(200, 2)):
    """
   <...> 
"""