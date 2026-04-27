#!/usr/bin/env python3
"""
Minimal spectral reconstruction framework:
- defines a spectral basis (Legendre polynomials)
- generates synthetic spectra
- produces synthetic photometric fluxes (U,B,V,R,I)
- reconstructs spectra from photometry via ridge regression
"""

import numpy as np
from numpy.polynomial.legendre import legval
from sklearn.linear_model import Ridge


# --------------------
# Spectral basis
# --------------------
def create_basis(n_basis: int, wavelengths: np.ndarray) -> np.ndarray:
    """
    Create a Legendre polynomial basis evaluated at given wavelengths.
    Returns a (n_basis, len(wavelengths)) array.
    """
    # Map wavelengths to [-1, 1]
    x = 2 * (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min()) - 1
    basis = np.vstack([legval(x, [0] * k + [1]) for k in range(n_basis)])
    return basis.astype(np.float64)


# --------------------
# Synthetic spectra
# --------------------
def generate_synthetic_spectra(
    n_samples: int,
    basis: np.ndarray,
    coeff_std: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic spectra and their coefficient vectors.
    Returns:
        spectra   : (n_samples, n_pixels)
        coeffs    : (n_samples, n_basis)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_basis = basis.shape[0]
    coeffs = rng.normal(scale=coeff_std, size=(n_samples, n_basis))
    spectra = coeffs @ basis
    return spectra, coeffs


# --------------------
# Filters
# --------------------
def create_filters(wavelengths: np.ndarray) -> list[dict]:
    """
    Create simple top-hat filters: U, B, V, R, I.
    Each filter is represented as a dictionary with name and response array.
    """
    filters = []

    # Filter definitions (center, width) in nm
    specs = [
        ("U", 365, 50),
        ("B", 445, 60),
        ("V", 551, 80),
        ("R", 658, 90),
        ("I", 806, 100),
    ]

    for name, center, width in specs:
        response = np.exp(-0.5 * ((wavelengths - center) / (width / 2.355)) ** 2)
        filters.append({"name": name, "response": response})

    return filters


# --------------------
# Photometry
# --------------------
def compute_photometry(
    spectra: np.ndarray,
    filters: list[dict],
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Compute synthetic fluxes for each spectrum through each filter.
    Returns (n_samples, n_filters).
    """
    delta = np.diff(wavelengths, append=wavelengths[-1])  # pixel width (nm)
    n_filters = len(filters)
    n_samples = spectra.shape[0]

    fluxes = np.empty((n_samples, n_filters), dtype=np.float64)

    for i, flt in enumerate(filters):
        resp = flt["response"]
        # Integrate spectrum * response over wavelength
        integrand = spectra * resp[None, :]  # broadcast over samples
        flux = np.sum(integrand * delta, axis=1)
        fluxes[:, i] = flux

    return fluxes


# --------------------
# Reconstruction
# --------------------
def build_transfer_matrix(
    basis: np.ndarray,
    filters: list[dict],
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Build the linear system matrix A such that flux = A @ coeffs.
    Returns matrix of shape (n_filters, n_basis).
    """
    delta = np.diff(wavelengths, append=wavelengths[-1])
    n_filters = len(filters)
    n_basis = basis.shape[0]
    A = np.empty((n_filters, n_basis), dtype=np.float64)

    for i, flt in enumerate(filters):
        resp = flt["response"]
        for k in range(n_basis):
            A[i, k] = np.sum(basis[k, :] * resp * delta)

    return A


def reconstruct_coeffs(
    fluxes: np.ndarray,
    A: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct coefficient vectors from photometric fluxes.
    Uses ridge regression: (A^T A + alpha I) c = A^T f.
    """
    AtA = A.T @ A
    reg = alpha * np.eye(AtA.shape[0], dtype=np.float64)
    AtF = A.T @ fluxes.T  # shape (n_basis, n_samples)

    coeffs_rec = np.linalg.solve(AtA + reg, AtF).T  # (n_samples, n_basis)
    return coeffs_rec


def reconstruct_spectra(
    coeffs: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """
    Compute spectra from coefficient vectors and basis.
    """
    return coeffs @ basis


# --------------------
# Demo
# --------------------
def main():
    # Wavelength grid (nm)
    wavelengths = np.linspace(350, 950, 3000, dtype=np.float64)

    # Spectral basis
    n_basis = 10
    basis = create_basis(n_basis, wavelengths)

    # Generate synthetic spectra
    n_samples = 20
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_samples, basis, coeff_std=1.0, rng=np.random.default_rng(42)
    )

    # Filters
    filters = create_filters(wavelengths)

    # Compute photometric fluxes
    fluxes = compute_photometry(spectra_true, filters, wavelengths)

    # Build transfer matrix
    A = build_transfer_matrix(basis, filters, wavelengths)

    # Reconstruct coefficients
    coeffs_rec = reconstruct_coeffs(fluxes, A, alpha=1.0)

    # Reconstruct spectra
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # Evaluation
    mse = np.mean((spectra_true - spectra_rec) ** 2)
    print(f"Mean squared error between true and reconstructed spectra: {mse:.4e}")

    # Show first spectrum comparison
    idx = 0
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectra_true[idx], label="True")
    plt.plot(wavelengths, spectra_rec[idx], label="Reconstructed", linestyle="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (arb. units)")
    plt.title(f"Spectrum {idx}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()