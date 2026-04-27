#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression


def generate_wavelength_grid(start=400, stop=800, num=1000):
    """Uniform wavelength grid in nm."""
    return np.linspace(start, stop, num)


def gaussian_profile(x, center, width, amplitude=1.0):
    """1‑D Gaussian profile."""
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def generate_basis(num_bases, wavelengths):
    """
    Generate a set of basis spectra.
    Each basis is a Gaussian with random centre and width.
    """
    rng = np.random.default_rng()
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), size=num_bases)
    widths = rng.uniform(10, 50, size=num_bases)
    amplitudes = rng.uniform(0.5, 1.5, size=num_bases)

    basis = np.array([gaussian_profile(wavelengths, c, w, a)
                      for c, w, a in zip(centers, widths, amplitudes)])
    return basis  # shape (num_bases, len(wavelengths))


def generate_synthetic_spectrum(coeffs, basis):
    """Linear combination of basis spectra with given coefficients."""
    return coeffs @ basis  # shape (len(wavelengths),)


def generate_filters(num_filters, wavelengths):
    """
    Create a set of synthetic photometric filters.
    Each filter is a Gaussian transmission curve.
    """
    rng = np.random.default_rng()
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), size=num_filters)
    widths = rng.uniform(20, 80, size=num_filters)
    filters = np.array([gaussian_profile(wavelengths, c, w) for c, w in zip(centers, widths)])
    # Normalise filter responses to unit area
    filters /= filters.sum(axis=1, keepdims=True)
    return filters  # shape (num_filters, len(wavelengths))


def compute_photometry(spectrum, filters, wavelength_step):
    """
    Integrate spectrum over each filter response.
    The integral is approximated by a weighted sum.
    """
    integrals = (spectrum * filters).sum(axis=1) * wavelength_step
    return integrals  # shape (num_filters,)


def build_design_matrix(basis, filters, wavelength_step):
    """
    Construct the linear model matrix relating basis coefficients
    to photometric fluxes: A_ij = integral(basis_j * filter_i).
    """
    num_filters, num_bases = filters.shape[0], basis.shape[0]
    A = np.zeros((num_filters, num_bases))
    for i in range(num_filters):
        A[i] = (basis * filters[i]).sum(axis=1) * wavelength_step
    return A  # shape (num_filters, num_bases)


def reconstruct_spectrum_from_photometry(photometry, basis, filters, wavelength_step):
    """
    Reconstruct the spectrum by fitting basis coefficients to photometry.
    Uses ordinary least squares via scikit‑learn LinearRegression.
    """
    A = build_design_matrix(basis, filters, wavelength_step)
    reg = LinearRegression(fit_intercept=False).fit(A, photometry)
    coeffs = reg.coef_
    return generate_synthetic_spectrum(coeffs, basis)


def main():
    # Wavelength grid
    wav = generate_wavelength_grid()
    dw = wav[1] - wav[0]

    # Basis spectra
    n_basis = 5
    basis = generate_basis(n_basis, wav)

    # True coefficients for synthetic spectrum
    rng = np.random.default_rng(42)
    true_coeffs = rng.normal(0, 1, size=n_basis)

    # Generate synthetic spectrum
    true_spectrum = generate_synthetic_spectrum(true_coeffs, basis)

    # Photometric filters
    n_filters = 6
    filters = generate_filters(n_filters, wav)

    # Compute synthetic photometry
    photometry = compute_photometry(true_spectrum, filters, dw)

    # Reconstruct spectrum
    recon_spectrum = reconstruct_spectrum_from_photometry(photometry, basis, filters, dw)

    # Evaluate reconstruction
    error = np.linalg.norm(true_spectrum - recon_spectrum) / np.linalg.norm(true_spectrum)
    print(f"Relative reconstruction error: {error:.4f}")

    # Optional: plot if desired
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(wav, true_spectrum, label="True Spectrum")
        plt.plot(wav, recon_spectrum, label="Reconstructed Spectrum", ls="--")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()