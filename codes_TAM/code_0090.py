#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.integrate import simps

# ------------------------------------------------------------
# 1. Spectral model
# ------------------------------------------------------------
def gaussian_basis(wavelengths, centers, widths):
    """
    Generate a set of Gaussian basis spectra.
    """
    basis = []
    for c, w in zip(centers, widths):
        g = np.exp(-0.5 * ((wavelengths - c) / w)**2)
        basis.append(g)
    return np.array(basis)  # shape (n_basis, n_wave)

def spectral_model(coeffs, basis):
    """
    Linear combination of basis spectra.
    """
    return np.dot(coeffs, basis)  # shape (n_wave,)

# ------------------------------------------------------------
# 2. Generate synthetic spectra
# ------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths, basis):
    """
    Generate random synthetic spectra.
    """
    n_basis = basis.shape[0]
    # Random coefficients: uniform [0, 1]
    coeffs = np.random.rand(n_samples, n_basis)
    spectra = coeffs @ basis.T  # shape (n_samples, n_wave)
    return spectra, coeffs

# ------------------------------------------------------------
# 3. Generate photometric data from spectra
# ------------------------------------------------------------
def gaussian_filter(wavelengths, center, width):
    """
    Simple Gaussian transmission curve.
    """
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def generate_filters(wavelengths, centers, widths):
    """
    Create filter transmission curves.
    """
    return np.array([gaussian_filter(wavelengths, c, w) for c, w in zip(centers, widths)])

def photometry_from_spectrum(spectrum, filters, wavelengths):
    """
    Compute fluxes through each filter for a single spectrum.
    """
    fluxes = []
    for filt in filters:
        # Weighted integral of spectrum * filter
        f = simps(spectrum * filt, wavelengths)
        fluxes.append(f)
    return np.array(fluxes)

def batch_photometry(spectra, filters, wavelengths):
    """
    Compute photometry for an array of spectra.
    """
    return np.array([photometry_from_spectrum(sp, filters, wavelengths) for sp in spectra])

# ------------------------------------------------------------
# 4. Reconstruct spectrum from photometry
# ------------------------------------------------------------
def train_coeff_reconstructor(photometry, coeffs):
    """
    Train linear regressor mapping photometry -> spectral coefficients.
    """
    lr = LinearRegression()
    lr.fit(photometry, coeffs)
    return lr

def reconstruct_spectrum(photometry_sample, lr, basis):
    """
    Predict coefficients from photometry and build spectrum.
    """
    coeff_pred = lr.predict(photometry_sample.reshape(1, -1))
    return spectral_model(coeff_pred.flatten(), basis)

# ------------------------------------------------------------
# 5. Demo
# ------------------------------------------------------------
def main():
    np.random.seed(42)

    # Wavelength grid (nm)
    wavelengths = np.linspace(300, 1000, 500)

    # Basis spectra
    basis_centers = [400, 500, 600, 700, 800]   # nm
    basis_widths  = [30, 40, 35, 45, 25]        # nm
    basis = gaussian_basis(wavelengths, basis_centers, basis_widths)

    # Generate training data
    n_train = 200
    spectra_train, coeffs_train = generate_synthetic_spectra(n_train, wavelengths, basis)

    # Filter set
    filter_centers = [350, 550, 750, 950]   # nm
    filter_widths  = [50, 50, 50, 50]       # nm
    filters = generate_filters(wavelengths, filter_centers, filter_widths)

    # Photometry for training set
    phot_train = batch_photometry(spectra_train, filters, wavelengths)

    # Train reconstructor
    lr = train_coeff_reconstructor(phot_train, coeffs_train)

    # Test on new synthetic spectra
    n_test = 10
    spectra_test, coeffs_test = generate_synthetic_spectra(n_test, wavelengths, basis)
    phot_test = batch_photometry(spectra_test, filters, wavelengths)

    # Reconstruct spectra
    spectra_rec = []
    for pt in phot_test:
        rec = reconstruct_spectrum(pt, lr, basis)
        spectra_rec.append(rec)
    spectra_rec = np.array(spectra_rec)

    # Display errors
    mae = np.mean(np.abs(spectra_rec - spectra_test), axis=1)
    print("Mean absolute error per test spectrum:", mae)

    # Example: plot first test and reconstructed spectrum
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(8,4))
        plt.plot(wavelengths, spectra_test[idx], label='True')
        plt.plot(wavelengths, spectra_rec[idx], '--', label='Reconstructed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux')
        plt.legend()
        plt.title('Spectrum Reconstruction Example')
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    main()