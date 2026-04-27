import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# 1. Define a simple spectral model
def spectral_model(wavelengths, coeffs):
    """Generate a spectrum given basis coefficients."""
    # basis: Gaussian bumps at 400, 500, 600 nm
    basis = []
    centers = [400, 500, 600]
    widths = [20, 30, 25]
    for c, w in zip(centers, widths):
        basis.append(np.exp(-((wavelengths - c) ** 2) / (2 * w ** 2)))
    basis = np.vstack(basis).T  # shape (Npix, 3)
    return basis @ coeffs

# 2. Generate synthetic spectra
def create_synthetic_spectra(num_spectra=10, num_pixels=300):
    wavelengths = np.linspace(350, 750, num_pixels)
    true_coeffs = np.random.uniform(0.5, 1.5, size=(num_spectra, 3))
    spectra = np.array([spectral_model(wavelengths, coeffs)
                        for coeff in true_coeffs])
    return wavelengths, spectra, true_coeffs

# 3. generate photometric data from synthetic spectra
def photometric_flux(
    wavelengths,
    spectra,
    filters,
    filter_wavelengths=None,
    normalize=False
):
    if filter_wavelengths is None:
        # assume same grid
        filter_wavelengths = wavelengths
    fluxes = []
    for filt in filters:
        # integrate spectrum × filter transmission
        prod = spectra * filt[:, None]
        # we assume filt is 1D array matching wavelengths
        # but here we will broadcast
        integ = simps(prod, wavelengths, axis=1)
        if normalize:
            integ /= np.sum(filt) ????; # <-- missing
        fluxes.append(integ)
    return np.array(flips).transpose()????...???



We must write final code. Need to correct errors. Let's design simple filters: maybe top-hat filters centered at 450,550,650 nm with width 50 nm. We'll create filter response arrays. Then compute photometric flux via integration. For reconstruction, we solve linear system: flux = F @ coeffs where F is matrix mapping coefficients to fluxes. Since spectral_model linear in coeffs, we can precompute basis integrated against filters to get design matrix. Then use Ridge regression to estimate coeffs from observed fluxes. Then reconstruct spectrum.

Let's implement functions: define_filters, generate_photometry, reconstruct_spectrum. Use synthetic data generation. Provide main block. Ensure no errors.

Let's code.assistantfinal code