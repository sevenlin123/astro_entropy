#!/usr/bin/env python3
import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

def build_gaussian_basis(wavelengths, n_basis):
    """Create a set of Gaussian basis functions."""
    gauss = gaussian(len(wavelengths), std=len(wavelengths)//(2*n_basis))
    basis = []
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    for c in centers:
        g = np.exp(-0.5*((wavelengths-c)/gauss.mean())**2)
        basis.append(g)
    return np.vstack(basis).T  # shape (n_wave, n_basis)

def generate_synthetic_spectra(n_samples, wavelengths, basis):
    """Generate spectra as linear combinations of basis functions."""
    rng = np.random.default_rng(42)
    coeffs = rng.normal(size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs  # shape (n_samples, n_wave), (n_samples, n_basis)

def build_filter_transmissions(wavelengths, n_filters):
    """Create simple top‑hat filter transmission curves."""
    filt = np.zeros((n_filters, len(wavelengths)))
    widths = (wavelengths[-1]-wavelengths[0])/(2*n_filters)
    for i in range(n_filters):
        center = wavelengths[0] + (i+0.5)*2*widths
        filt[i, (wavelengths>=center-widths) & (wavelengths<=center+widths)] = 1.0
    return filt  # shape (n_filters, n_wave)

def photometry_from_spectra(spectra, filters, wavelengths):
    """Integrate spectra through each filter to obtain fluxes."""
    dx = np.gradient(wavelengths)
    fluxes = (spectra @ filters.T) * dx  # shape (n_samples, n_filters)
    return fluxes

def reconstruct_spectrum_from_photometry(fluxes, basis, filters, alpha=0.1):
    """Recover basis coefficients using ridge regression."""
    # Project basis into photometric space
    M = filters @ basis  # shape (n_filters, n_basis)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(M, fluxes.T)
    coeffs_est = model.coef_.T  # shape (n_samples, n_basis)
    spectra_rec = coeffs_est @ basis.T
    return spectra_rec, coeffs_est

def main():
    # Wavelength grid
    wavelengths = np.linspace(400, 800, 200)  # nm

    # Build basis and synthetic spectra
    basis = build_gaussian_basis(wavelengths, n_basis=5)
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples=10,
                                                          wavelengths=wavelengths,
                                                          basis=basis)

    # Build filters and generate photometry
    filters = build_filter_transmissions(wavelengths, n_filters=3)
    fluxes = photometry_from_spectra(spectra_true, filters, wavelengths)

    # Reconstruct spectra
    spectra_rec, coeffs_est = reconstruct_spectrum_from_photometry(
        fluxes, basis, filters, alpha=0.01)

    # Display results
    print("True coefficients:\n", coeffs_true[:3])
    print("\nEstimated coefficients:\n", coeffs_est[:3])
    print("\nReconstruction error (RMSE):",
          np.sqrt(np.mean((spectra_true - spectra_rec)**2)))

if __name__ == "__main__":
    main()