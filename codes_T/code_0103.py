#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# -------------------- Spectral model ---------------------------------
def wavelength_grid(start=4000, stop=7000, num=1000):
    """Create a wavelength grid in Å."""
    return np.linspace(start, stop, num)

def gaussian(wl, center, sigma, amplitude=1.0):
    """One-dimensional Gaussian profile."""
    return amplitude * np.exp(-0.5 * ((wl - center) / sigma)**2)

def generate_basis_functions(num_basis, wl):
    """
    Generate a set of basis functions (Gaussian peaks).
    Returns an array of shape (num_basis, len(wl)).
    """
    centers = np.linspace(wl[0], wl[-1], num_basis)
    sigmas = 50 + 10 * np.random.randn(num_basis)
    sigmas = np.abs(sigmas) + 20
    basis = np.vstack([gaussian(wl, c, s) for c, s in zip(centers, sigmas)])
    return basis

# -------------------- Synthetic spectra ---------------------------------
def generate_synthetic_spectra(num_spectra, basis):
    """
    Generate synthetic spectra as random linear combinations of the basis.
    Returns an array of shape (num_spectra, len(basis[0])).
    """
    coeffs = np.random.rand(num_spectra, basis.shape[0])
    spectra = coeffs @ basis
    return spectra, coeffs

# -------------------- Filters ---------------------------------
def generate_filters(num_filters, wl):
    """
    Generate simple top-hat filter transmission curves.
    Returns a list of arrays of length len(wl).
    """
    filters = []
    band_width = (wl[-1] - wl[0]) / (num_filters * 1.5)
    for i in range(num_filters):
        low = wl[0] + i * band_width
        high = low + band_width
        trans = np.where((wl >= low) & (wl <= high), 1.0, 0.0)
        filters.append(trans)
    return filters

# -------------------- Photometry ---------------------------------
def compute_photometry(spectra, filters, wl):
    """
    Compute synthetic photometry by integrating spectrum * filter over wavelength.
    Returns an array of shape (len(spectra), len(filters)).
    """
    phot = np.zeros((spectra.shape[0], len(filters)))
    for j, filt in enumerate(filters):
        # integrate spectrum * transmission
        integrand = spectra * filt
        phot[:, j] = np.trapz(integrand, wl, axis=1)
    return phot

# -------------------- Reconstruction ---------------------------------
def reconstruct_spectrum_from_photometry(photometry, filters, basis, wl):
    """
    Reconstruct spectra from photometric data using linear regression.
    Returns reconstructed spectra and estimated coefficients.
    """
    # Build design matrix: filter integrals of each basis function
    # Each column corresponds to one basis, each row to one filter
    design = np.zeros((len(filters), basis.shape[0]))
    for i, filt in enumerate(filters):
        # integrate each basis * filter
        integrands = basis.T * filt   # shape (num_basis, len(wl))
        design[i, :] = np.trapz(integrands, wl, axis=1)
    # Fit linear model
    lr = LinearRegression(fit_intercept=False)
    lr.fit(design, photometry.T)          # photometry.T shape (num_filters, num_spectra)
    coeffs_est = lr.coef_.T                # shape (num_spectra, num_basis)
    spectra_rec = coeffs_est @ basis      # reconstruct spectra
    return spectra_rec, coeffs_est

# -------------------- Main ---------------------------------
if __name__ == "__main__":
    # Parameters
    N_WL = 1500
    N_BASIS = 10
    N_SPECTRA = 5
    N_FILTERS = 6

    # Generate wavelength grid
    wl = wavelength_grid(num=N_WL)

    # Basis functions
    basis = generate_basis_functions(N_BASIS, wl)

    # Synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(N_SPECTRA, basis)

    # Filters
    filters = generate_filters(N_FILTERS, wl)

    # Photometry
    photometry = compute_photometry(spectra_true, filters, wl)

    # Reconstruction
    spectra_rec, coeffs_est = reconstruct_spectrum_from_photometry(
        photometry, filters, basis, wl
    )

    # Display results
    print("True coefficients shape:", coeffs_true.shape)
    print("Estimated coefficients shape:", coeffs_est.shape)
    print("Reconstructed spectra shape:", spectra_rec.shape)
    # Compare first spectrum
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(wl, spectra_true[0], label='True')
    plt.plot(wl, spectra_rec[0], '--', label='Reconstructed')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()
    plt.show()