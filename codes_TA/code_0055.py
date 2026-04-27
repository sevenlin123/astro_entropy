#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression

def wavelength_grid(npts=1000, lam_min=400.0, lam_max=800.0):
    """
    Return a linear wavelength grid in nanometers.
    """
    return np.linspace(lam_min, lam_max, npts)

def basis_functions(lam):
    """
    Construct basis functions (constant, linear, quadratic) evaluated at `lam`.
    Returns an array of shape (len(lam), 3).
    """
    ones = np.ones_like(lam)
    lin = lam - lam.mean()
    quad = (lam - lam.mean())**2
    return np.vstack((ones, lin, quad)).T

def spectral_model(coeffs, lam):
    """
    Compute a synthetic spectrum as a linear combination of basis functions.
    """
    B = basis_functions(lam)
    return B @ coeffs

def filter_gaussian(lam, center, width):
    """
    Gaussian filter transmission curve.
    """
    return np.exp(-0.5 * ((lam - center) / width)**2)

def integrate_flux(flux, transmission, lam):
    """
    Integrate flux weighted by a filter transmission curve.
    """
    return np.trapz(flux * transmission, lam) / np.trapz(transmission, lam)

def generate_filters(lam, centers=[500.0, 600.0], widths=[20.0, 25.0]):
    """
    Generate a list of filter transmission curves.
    """
    filters = []
    for c, w in zip(centers, widths):
        filters.append(filter_gaussian(lam, c, w))
    return filters

def compute_photometry(spectrum, filters, lam):
    """
    Compute photometric measurements by integrating the spectrum through each filter.
    """
    return np.array([integrate_flux(spectrum, filt, lam) for filt in filters])

def generate_synthetic_dataset(seed=42):
    """
    Generate synthetic spectrum and corresponding photometric data.
    Returns:
        lam: wavelength grid
        true_coeffs: ground truth basis coefficients
        spectrum: synthetic flux array
        photometry: simulated photometric measurements
        filters: list of filter transmission curves
    """
    rng = np.random.default_rng(seed)
    lam = wavelength_grid()
    true_coeffs = rng.uniform(-1.0, 1.0, size=3)
    spectrum = spectral_model(true_coeffs, lam)
    filters = generate_filters(lam)
    photometry = compute_photometry(spectrum, filters, lam)
    return lam, true_coeffs, spectrum, photometry, filters

def reconstruct_spectrum(photometry, filters, lam):
    """
    Reconstruct basis coefficients from photometric data using linear regression.
    Returns:
        est_coeffs: estimated basis coefficients
        reconstructed_flux: flux array reconstructed from estimated coeffs
    """
    # Build design matrix A where A[j, k] = integral of basis_k through filter_j
    B = basis_functions(lam)
    n_filters = len(filters)
    n_basis = B.shape[1]
    A = np.empty((n_filters, n_basis))
    for j, filt in enumerate(filters):
        for k in range(n_basis):
            A[j, k] = integrate_flux(B[:, k], filt, lam)

    # Fit linear regression (no intercept)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A, photometry)
    est_coeffs = reg.coef_
    reconstructed_flux = B @ est_coeffs
    return est_coeffs, reconstructed_flux

def main():
    lam, true_coeffs, spectrum, photometry, filters = generate_synthetic_dataset()
    est_coeffs, recon_flux = reconstruct_spectrum(photometry, filters, lam)

    # Evaluate reconstruction
    mse = np.mean((spectrum - recon_flux)**2)
    print("True coefficients :", true_coeffs)
    print("Estimated coeffs   :", est_coeffs)
    print("MSE of reconstructed spectrum :", mse)

if __name__ == "__main__":
    main()