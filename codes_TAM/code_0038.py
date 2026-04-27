#!/usr/bin/env python3
# Minimal spectral reconstruction from photometry

import numpy as np
from scipy.signal import gaussian


def create_gaussian_basis(num_bases, wavelengths):
    """
    Create a set of Gaussian basis functions with different centres.
    """
    rng = np.random.default_rng(seed=42)
    centres = rng.uniform(wavelengths.min(), wavelengths.max(), size=num_bases)
    widths = rng.uniform((wavelengths.max() - wavelengths.min()) / 20,
                         (wavelengths.max() - wavelengths.min()) / 10,
                         size=num_bases)
    basis = np.array([gaussian(len(wavelengths), std=int(width),
                               center=int((centre - wavelengths.min()) /
                                          (wavelengths[1] - wavelengths[0])))
                      for centre, width in zip(centres, widths)])
    # Normalise each basis to unit integral
    dw = wavelengths[1] - wavelengths[0]
    basis = basis / (basis.sum(axis=1, keepdims=True) * dw)
    return basis  # shape (num_bases, n_wavelengths)


def create_gaussian_filters(num_filters, wavelengths):
    """
    Create Gaussian filter curves centred at different wavelengths.
    """
    rng = np.random.default_rng(seed=24)
    centres = rng.uniform(wavelengths.min() + 50, wavelengths.max() - 50,
                          size=num_filters)
    widths = rng.uniform(30, 70, size=num_filters)
    filters = np.array([gaussian(len(wavelengths), std=int(width),
                                 center=int((centre - wavelengths.min()) /
                                            (wavelengths[1] - wavelengths[0])))
                        for centre, width in zip(centres, widths)])
    # Normalise each filter to unit area
    dw = wavelengths[1] - wavelengths[0]
    filters = filters / (filters.sum(axis=1, keepdims=True) * dw)
    return filters  # shape (num_filters, n_wavelengths)


def generate_synthetic_spectra(num_spectra, basis, coeff_bounds=(0, 1)):
    """
    Generate synthetic spectra as linear combinations of basis functions.
    """
    rng = np.random.default_rng(seed=12)
    coeffs = rng.uniform(coeff_bounds[0], coeff_bounds[1],
                         size=(len(basis), num_spectra))
    spectra = coeffs.T @ basis  # shape (num_spectra, n_wavelengths)
    return spectra, coeffs  # return true coefficients for comparison


def compute_photometry(spectra, filters, wavelengths):
    """
    Integrate each spectrum through each filter to obtain photometric fluxes.
    """
    dw = wavelengths[1] - wavelengths[0]
    # filters: (n_filters, n_wavelengths)
    # spectra: (n_spectra, n_wavelengths)
    photometry = spectra @ filters.T * dw  # shape (n_spectra, n_filters)
    return photometry.T  # shape (n_filters, n_spectra)


def reconstruct_coefficients(photometry, basis, filters, wavelengths):
    """
    Reconstruct basis coefficients from photometry by solving a linear system.
    """
    dw = wavelengths[1] - wavelengths[0]
    # Build the system matrix A: A_ij = ∫ filter_i * basis_j dλ
    A = filters @ basis.T * dw  # shape (n_filters, n_bases)
    # Solve for coefficients for each spectrum
    coeffs, *_ = np.linalg.lstsq(A, photometry, rcond=None)  # shape (n_bases, n_spectra)
    return coeffs


def reconstruct_spectra(coeffs, basis):
    """
    Reconstruct spectra from basis coefficients.
    """
    # coeffs: (n_bases, n_spectra)
    spectra = coeffs.T @ basis  # shape (n_spectra, n_wavelengths)
    return spectra


if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(400, 800, 400)  # nm

    # Create basis and filters
    basis = create_gaussian_basis(num_bases=5, wavelengths=wav)
    filters = create_gaussian_filters(num_filters=4, wavelengths=wav)

    # Generate synthetic spectra
    n_spec = 10
    spectra_true, coeffs_true = generate_synthetic_spectra(n_spec, basis)

    # Compute photometry
    phot = compute_photometry(spectra_true, filters, wav)

    # Reconstruct spectra
    coeffs_rec = reconstruct_coefficients(phot, basis, filters, wav)
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # Evaluate reconstruction quality
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared error of reconstructed spectra: {mse:.6f}")

    # Optional: compare first spectrum visually (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(8, 4))
        plt.plot(wav, spectra_true[idx], label="True")
        plt.plot(wav, spectra_rec[idx], '--', label="Reconstructed")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux")
        plt.title("Spectrum Reconstruction Example")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass