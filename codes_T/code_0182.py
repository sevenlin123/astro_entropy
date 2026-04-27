#!/usr/bin/env python3
# Minimal spectral reconstruction framework

import numpy as np
from scipy.special import erf
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# Spectral model
# --------------------------------------------------------------------

def wavelength_grid(start=350, stop=950, num=1000):
    """
    Return a monotonic wavelength grid (in nm).
    """
    return np.linspace(start, stop, num)

def gaussian(x, center, width):
    """
    Normalized Gaussian function.
    """
    return np.exp(-((x - center) ** 2) / (2 * width ** 2))

def generate_basis_spectra(n_basis, wavelengths):
    """
    Generate n_basis Gaussian basis spectra over the wavelength grid.
    Each basis has a random centre and width within reasonable limits.
    """
    rng = np.random.default_rng(seed=42)
    basis = []
    for _ in range(n_basis):
        center = rng.uniform(400, 800)          # nm
        width  = rng.uniform(20, 80)            # nm
        basis.append(gaussian(wavelengths, center, width))
    return np.vstack(basis)  # shape (n_basis, len(wavelengths))

def generate_coefficients(n_samples, n_basis):
    """
    Generate random linear combination coefficients for synthetic spectra.
    """
    rng = np.random.default_rng(seed=123)
    coeffs = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_basis))
    return coeffs

def generate_synthetic_spectra(coeffs, basis):
    """
    Construct synthetic spectra as linear combinations of basis spectra.
    """
    return coeffs @ basis  # shape (n_samples, n_wavelengths)

# --------------------------------------------------------------------
# Photometric filter definition
# --------------------------------------------------------------------

def rectangular_filter(wavelengths, center, width):
    """
    Simple rectangular filter profile.
    """
    return ((wavelengths >= center - width / 2) &
            (wavelengths <= center + width / 2)).astype(float)

def generate_filters():
    """
    Define a set of three synthetic broad-band filters (U, B, V).
    Returns a list of filter transmission vectors aligned with the wavelength grid.
    """
    wavelengths = wavelength_grid()
    filters = {
        'U': rectangular_filter(wavelengths, 360, 80),
        'B': rectangular_filter(wavelengths, 440, 80),
        'V': rectangular_filter(wavelengths, 550, 80),
    }
    return filters, wavelengths

# --------------------------------------------------------------------
# Photometry computation
# --------------------------------------------------------------------

def compute_photometry(spectra, filters, wavelengths):
    """
    Integrate each spectrum through each filter.
    Returns an array of shape (n_samples, n_filters).
    """
    phot = []
    for name in sorted(filters.keys()):
        filt = filters[name]
        # Simple trapezoidal integration over the wavelength grid
        integrand = spectra * filt
        flux = np.trapz(integrand, wavelengths, axis=1)
        norm = np.trapz(filt, wavelengths)  # normalize by filter width
        phot.append(flux / norm)
    return np.column_stack(phot)  # shape (n_samples, n_filters)

# --------------------------------------------------------------------
# Filter response matrix
# --------------------------------------------------------------------

def compute_filter_response(basis, filters, wavelengths):
    """
    Compute the response of each basis spectrum in each filter.
    Returns a matrix of shape (n_filters, n_basis).
    """
    responses = []
    for name in sorted(filters.keys()):
        filt = filters[name]
        resp = np.array([np.trapz(b * filt, wavelengths) / np.trapz(filt, wavelengths)
                         for b in basis])
        responses.append(resp)
    return np.vstack(responses)  # shape (n_filters, n_basis)

# --------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------

def reconstruct_from_photometry(phot_obs, filter_response, basis):
    """
    Given observed photometry (vector), filter response matrix, and basis spectra,
    reconstruct the underlying spectrum by fitting coefficients via linear regression.
    Returns the reconstructed spectrum vector.
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(filter_response.T, phot_obs)
    coeffs = reg.coef_
    recon_spectrum = coeffs @ basis
    return recon_spectrum

# --------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Generate basis and synthetic data
    wavelengths = wavelength_grid()
    n_basis = 5
    basis = generate_basis_spectra(n_basis, wavelengths)

    n_samples = 10
    coeffs = generate_coefficients(n_samples, n_basis)
    spectra = generate_synthetic_spectra(coeffs, basis)

    # Define filters
    filters, wavelengths = generate_filters()

    # Compute photometry for all samples
    photometry = compute_photometry(spectra, filters, wavelengths)

    # Compute filter response matrix for the basis
    filter_response = compute_filter_response(basis, filters, wavelengths)

    # Reconstruct first spectrum as demonstration
    idx = 0
    recon_spectrum = reconstruct_from_photometry(
        photometry[idx],
        filter_response,
        basis
    )

    # Print comparison
    print("True spectrum (first sample):")
    print(spectra[idx][:5])   # show first 5 values
    print("\nReconstructed spectrum:")
    print(recon_spectrum[:5]) # show first 5 values
    print("\nReconstruction error (RMSE):")
    rmse = np.sqrt(np.mean((spectra[idx] - recon_spectrum)**2))
    print(rmse)