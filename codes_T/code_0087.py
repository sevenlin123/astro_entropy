#!/usr/bin/env python3
import numpy as np
from scipy import signal

# ----------------- Spectral model -----------------
def generate_basis(wavelength, n_bases=3):
    """
    Create simple orthogonal basis spectra:
        0: constant
        1: linear trend
        2: sinusoidal
    """
    basis = []
    basis.append(np.ones_like(wavelength))
    basis.append((wavelength - wavelength.mean()) / wavelength.ptp())
    basis.append(np.sin(4 * np.pi * (wavelength - wavelength.min()) / wavelength.ptp()))
    return np.vstack(basis)  # shape (n_bases, n_wavelength)

def synthesize_spectrum(coeffs, basis):
    """Linear combination of basis spectra."""
    return np.dot(coeffs, basis)

# ----------------- Photometric system -----------------
def generate_filters(n_filters, wavelength, width=30):
    """
    Create Gaussian filters centered at equally spaced wavelengths.
    Returns list of throughput arrays.
    """
    centers = np.linspace(wavelength.min()+0.2*wavelength.ptp(),
                          wavelength.max()-0.2*wavelength.ptp(),
                          n_filters)
    filters = []
    for c in centers:
        filt = signal.gaussian(len(wavelength), std=width*len(wavelength)/wavelength.ptp())
        filt *= np.exp(-(wavelength - c)**2 / (2*(width**2)))
        filt /= filt.sum()  # normalize
        filters.append(filt)
    return filters

def photometry_from_spectrum(spectrum, filters):
    """Integrate spectrum over each filter (trapezoidal rule)."""
    phot = []
    for f in filters:
        phot.append(np.trapz(spectrum * f, dx=1))  # assume uniform spacing
    return np.array(phot)

# ----------------- Reconstruction -----------------
def reconstruct_spectrum(photometry, filters, basis):
    """
    Estimate coefficients via least squares:
        photometry ≈ A @ coeffs
    where A[i,j] = ∫ (basis[j] * filter[i]) dλ
    """
    n_bases = basis.shape[0]
    n_filters = len(filters)
    A = np.zeros((n_filters, n_bases))
    for i, f in enumerate(filters):
        for j in range(n_bases):
            A[i, j] = np.trapz(basis[j] * f, dx=1)
    coeffs, *_ = np.linalg.lstsq(A, photometry, rcond=None)
    return coeffs

# ----------------- Synthetic experiment -----------------
def main():
    np.random.seed(0)
    wavelength = np.linspace(400, 800, 200)   # nm
    basis = generate_basis(wavelength, n_bases=3)
    true_coeffs = np.array([2.0, 0.5, 1.0])   # arbitrary
    true_spectrum = synthesize_spectrum(true_coeffs, basis)

    filters = generate_filters(n_filters=5, wavelength=wavelength, width=15)
    phot = photometry_from_spectrum(true_spectrum, filters)

    est_coeffs = reconstruct_spectrum(phot, filters, basis)
    reconstructed = synthesize_spectrum(est_coeffs, basis)

    print("True coeffs:", true_coeffs)
    print("Estimated coeffs:", est_coeffs)
    print("Reconstruction error (L2):", np.linalg.norm(true_spectrum - reconstructed))

if __name__ == "__main__":
    main()