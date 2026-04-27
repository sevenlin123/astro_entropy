#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression


def generate_wavelengths(start=350, stop=800, n=1000):
    """Create an evenly spaced wavelength grid (in nm)."""
    return np.linspace(start, stop, n)


def basis_functions(wavelengths, n_basis=5):
    """
    Generate a set of basis functions evaluated on the wavelength grid.
    The basis consists of orthonormalized polynomials up to order n_basis-1.
    """
    # Normalize wavelengths to [-1, 1]
    x = (wavelengths - wavelengths.mean()) / (wavelengths.ptp() / 2)
    basis = [x**i for i in range(n_basis)]
    return basis


def generate_synthetic_spectra(num_spectra, wavelengths, basis):
    """
    Create synthetic spectra as linear combinations of the basis functions.
    Returns:
        coeffs: (num_spectra, n_basis) array of coefficients
        spectra: (num_spectra, n_wavelengths) array of fluxes
    """
    n_basis = len(basis)
    coeffs = np.random.randn(num_spectra, n_basis)
    spectra = np.array([coeffs[i] @ np.vstack(basis) for i in range(num_spectra)])
    return coeffs, spectra


def gaussian_filter(wavelengths, center, width):
    """Return a Gaussian filter transmission curve."""
    filt = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
    # Normalize so that integral ≈ 1
    filt /= np.trapz(filt, wavelengths)
    return filt


def generate_filters(wavelengths, centers, width):
    """Generate a set of Gaussian filters."""
    return [gaussian_filter(wavelengths, c, width) for c in centers]


def compute_photometry(spectra, filters):
    """
    Integrate each spectrum through each filter.
    Returns an array of shape (num_spectra, num_filters).
    """
    phot = np.array([
        [np.trapz(s * f, wavelengths) for f in filters]
        for s in spectra
    ])
    return phot


def build_filter_matrix(filters, basis, wavelengths):
    """
    Construct matrix A where A[i, j] = ∫ basis_j(λ) * filter_i(λ) dλ.
    """
    n_filters = len(filters)
    n_basis = len(basis)
    A = np.empty((n_filters, n_basis))
    for i, f in enumerate(filters):
        for j, b in enumerate(basis):
            A[i, j] = np.trapz(b * f, wavelengths)
    return A


def reconstruct_spectrum(photometry, A):
    """
    Recover coefficients for each spectrum by least‑squares fitting.
    Returns an array of shape (num_spectra, n_basis).
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A, photometry.T)  # shape: (n_filters, n_spectra)
    coeffs_rec = reg.coef_.T  # transpose to (n_spectra, n_basis)
    return coeffs_rec


def reconstruct_fluxes(coeffs_rec, basis):
    """Build reconstructed spectra from recovered coefficients."""
    return np.array([c @ np.vstack(basis) for c in coeffs_rec])


# --------------------------- MAIN -------------------------------------------------
if __name__ == "__main__":
    # Parameters
    num_spectra = 10
    n_basis = 5
    filter_centers = [400, 500, 600]  # nm
    filter_width = 30  # nm

    # Step 1: Create wavelength grid
    wavelengths = generate_wavelengths()

    # Step 2: Define basis functions
    basis = basis_functions(wavelengths, n_basis=n_basis)

    # Step 3: Generate synthetic spectra
    true_coeffs, spectra = generate_synthetic_spectra(
        num_spectra, wavelengths, basis
    )

    # Step 4: Generate filters and compute photometry
    filters = generate_filters(wavelengths, filter_centers, filter_width)
    phot = compute_photometry(spectra, filters)

    # Add Gaussian noise to photometric measurements
    phot += 0.01 * np.random.randn(*phot.shape)

    # Step 5: Build filter matrix for linear regression
    A = build_filter_matrix(filters, basis, wavelengths)

    # Step 6: Reconstruct coefficients from photometry
    rec_coeffs = reconstruct_spectrum(phot, A)

    # Step 7: Reconstruct spectra from recovered coefficients
    spectra_rec = reconstruct_fluxes(rec_coeffs, basis)

    # Display results for the first spectrum
    idx = 0
    print("True coefficients:", true_coeffs[idx])
    print("Recovered coefficients:", rec_coeffs[idx])

    # Compute reconstruction error
    err = np.linalg.norm(spectra[idx] - spectra_rec[idx]) / np.linalg.norm(spectra[idx])
    print(f"Relative L2 reconstruction error for spectrum {idx}: {err:.4f}")