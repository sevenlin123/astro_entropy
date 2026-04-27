#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework.
"""

import numpy as np
from scipy.stats import uniform


def wavelength_grid(start=400, stop=800, num=1000):
    """Return a wavelength array in nanometers."""
    return np.linspace(start, stop, num)


def gaussian_basis(wavelengths, centers, widths):
    """Return an array of Gaussian basis functions evaluated on wavelengths."""
    n_centers = len(centers)
    basis = np.exp(
        -0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :]) ** 2
    )
    return basis  # shape (n_wavelengths, n_bases)


def generate_synthetic_spectra(coeffs, basis):
    """Linear combination of basis functions with given coefficients."""
    return coeffs @ basis.T  # shape (n_spectra, n_wavelengths)


def random_bandpasses(n_bands, wavelengths, width=40):
    """
    Generate random top-hat bandpasses.
    Each bandpass is a transmission curve equal to 1 inside
    [center - width/2, center + width/2] and 0 outside.
    """
    min_wave, max_wave = wavelengths[0], wavelengths[-1]
    centers = uniform.rvs(loc=min_wave + width / 2,
                          scale=max_wave - min_wave - width,
                          size=n_bands)
    transmissions = np.zeros((len(wavelengths), n_bands))
    for i, cen in enumerate(centers):
        low, high = cen - width / 2, cen + width / 2
        mask = (wavelengths >= low) & (wavelengths <= high)
        transmissions[mask, i] = 1.0
    return transmissions  # shape (n_wavelengths, n_bands)


def compute_photometry(spectra, transmissions, wavelengths):
    """
    Integrate spectra over each transmission curve.
    spectra: (n_spectra, n_wavelengths)
    transmissions: (n_wavelengths, n_bands)
    """
    dw = np.gradient(wavelengths)  # differential element
    # Multiply spectra with transmissions and integrate over wavelength
    photometry = (spectra[:, :, None] * transmissions[None, :, :]).sum(axis=1) * dw.mean()
    return photometry  # shape (n_spectra, n_bands)


def build_forward_matrix(basis, transmissions, wavelengths):
    """
    Build matrix A such that photometry = A @ coeffs.
    basis: (n_wavelengths, n_bases)
    transmissions: (n_wavelengths, n_bands)
    """
    dw = np.gradient(wavelengths)
    # For each band, integrate basis functions over that band
    A = (basis.T @ transmissions) * dw.mean()  # shape (n_bases, n_bands)
    return A.T  # shape (n_bands, n_bases)


def reconstruct_coefficients(photometry, forward_matrix):
    """Solve linear least‑squares for basis coefficients."""
    # Solve photometry = forward_matrix @ coeffs
    coeffs, *_ = np.linalg.lstsq(forward_matrix, photometry.T, rcond=None)
    return coeffs.T  # shape (n_spectra, n_bases)


def main():
    rng = np.random.default_rng(seed=42)

    # 1. Define wavelength grid
    wl = wavelength_grid()

    # 2. Define spectral model: Gaussian basis functions
    n_bases = 10
    centers = np.linspace(420, 780, n_bases) + rng.normal(scale=10, size=n_bases)
    widths = np.full(n_bases, 30.0)
    basis = gaussian_basis(wl, centers, widths)

    # 3. Generate synthetic spectra
    n_spectra = 5
    coeff_low, coeff_high = -1.0, 1.0
    coeffs_true = rng.uniform(coeff_low, coeff_high, size=(n_spectra, n_bases))
    spectra_true = generate_synthetic_spectra(coeffs_true, basis)

    # 4. Generate photometric data
    n_bands = 7
    transmissions = random_bandpasses(n_bands, wl, width=40)
    photometry = compute_photometry(spectra_true, transmissions, wl)

    # 5. Reconstruct spectra
    A = build_forward_matrix(basis, transmissions, wl)  # shape (n_bands, n_bases)
    coeffs_est = reconstruct_coefficients(photometry, A)  # shape (n_spectra, n_bases)
    spectra_est = generate_synthetic_spectra(coeffs_est, basis)

    # 6. Compare true vs reconstructed spectra
    mse = ((spectra_true - spectra_est) ** 2).mean(axis=1)
    for i, err in enumerate(mse):
        print(f"Spectrum {i:02d}: MSE={err:.4e}")

    # Optional: show first spectrum comparison (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.plot(wl, spectra_true[idx], label="True")
        plt.plot(wl, spectra_est[idx], "--", label="Reconstructed")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux")
        plt.title(f"Spectrum {idx}")
        plt.legend()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()