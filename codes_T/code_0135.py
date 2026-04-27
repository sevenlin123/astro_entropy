#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge


def generate_basis(n_basis, n_points, lam_min=400, lam_max=1000):
    """
    Generate synthetic basis spectra as Gaussian peaks.
    """
    wavelengths = np.linspace(lam_min, lam_max, n_points)
    basis = np.zeros((n_basis, n_points))
    centers = np.linspace(lam_min, lam_max, n_basis)
    widths = (lam_max - lam_min) / (n_basis * 4)

    for i in range(n_basis):
        basis[i] = np.exp(-0.5 * ((wavelengths - centers[i]) / widths) ** 2)
    return wavelengths, basis


def create_synthetic_spectrum(coeffs, basis):
    """
    Linear combination of basis spectra.
    """
    return coeffs @ basis  # shape (n_points,)


def generate_filter_responses(n_filters, n_points, lam_min=400, lam_max=1000):
    """
    Simple Gaussian filters.
    """
    wavelengths = np.linspace(lam_min, lam_max, n_points)
    filters = np.zeros((n_filters, n_points))
    centers = np.linspace(lam_min, lam_max, n_filters)
    widths = (lam_max - lam_min) / (n_filters * 6)

    for j in range(n_filters):
        filters[j] = np.exp(-0.5 * ((wavelengths - centers[j]) / widths) ** 2)
    return wavelengths, filters


def compute_filter_matrix(basis, filters):
    """
    Compute M_{j,i} = \int B_i(λ) * T_j(λ) dλ
    """
    n_filters, n_points = filters.shape
    n_basis = basis.shape[0]
    M = np.zeros((n_filters, n_basis))

    for j in range(n_filters):
        for i in range(n_basis):
            M[j, i] = simps(basis[i] * filters[j], dx=filters.shape[1])
    return M


def generate_photometric_data(spectrum, filters):
    """
    Integrate spectrum over filter responses.
    """
    n_filters = filters.shape[0]
    phot = np.zeros(n_filters)
    for j in range(n_filters):
        phot[j] = simps(spectrum * filters[j], dx=filters.shape[1])
    return phot


def reconstruct_coefficients(photometry, filter_matrix, alpha=1e-3):
    """
    Solve M @ c = p with Tikhonov regularisation.
    """
    M = filter_matrix
    AtA = M.T @ M
    Atb = M.T @ photometry
    c = np.linalg.solve(AtA + alpha * np.eye(AtA.shape[0]), Atb)
    return c


def reconstruct_spectrum(coeffs, basis):
    """
    Reconstruct full spectrum from coefficients.
    """
    return coeffs @ basis


def main():
    np.random.seed(0)

    # Parameters
    n_basis = 5
    n_filters = 3
    n_points = 200

    # Basis spectra
    wav, basis = generate_basis(n_basis, n_points)

    # Coefficients for synthetic spectrum
    true_coeffs = np.random.rand(n_basis)
    true_spectrum = create_synthetic_spectrum(true_coeffs, basis)

    # Filter responses
    _, filters = generate_filter_responses(n_filters, n_points)

    # Photometric data
    photometry = generate_photometric_data(true_spectrum, filters)

    # Filter matrix
    filter_matrix = compute_filter_matrix(basis, filters)

    # Reconstruction
    recovered_coeffs = reconstruct_coefficients(photometry, filter_matrix, alpha=1e-2)
    recovered_spectrum = reconstruct_spectrum(recovered_coeffs, basis)

    # Print results
    print("True coefficients:     ", true_coeffs)
    print("Recovered coefficients:", recovered_coeffs)
    print("\nTrue spectrum vs Recovered spectrum (first 10 points):")
    print(np.vstack([true_spectrum[:10], recovered_spectrum[:10]]))


if __name__ == "__main__":
    main()