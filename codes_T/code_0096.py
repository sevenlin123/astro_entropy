import numpy as np
from numpy.linalg import lstsq, pinv
from scipy.stats import norm

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def gaussian(x, mu, sigma):
    """One-dimensional Gaussian."""
    return norm.pdf(x, loc=mu, scale=sigma)

def get_basis_functions(n_basis, wavelengths):
    """Generate a set of Gaussian basis functions."""
    np.random.seed(42)
    mus = np.linspace(450, 750, n_basis)
    sigmas = np.full(n_basis, 30.0)  # fixed width
    basis = np.array([gaussian(wavelengths, mu, sigma) for mu, sigma in zip(mus, sigmas)])
    return basis  # shape: (n_basis, n_wavelengths)

def get_filter_transmissions(n_filters, wavelengths):
    """Generate simple top-hat or Gaussian filter profiles."""
    np.random.seed(24)
    mus = np.linspace(460, 740, n_filters)
    widths = np.full(n_filters, 60.0)  # filter width
    filters = np.array([gaussian(wavelengths, mu, width) for mu, width in zip(mus, widths)])
    return filters  # shape: (n_filters, n_wavelengths)

# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis_funcs):
    """Create synthetic spectra as linear combinations of basis functions."""
    np.random.seed(7)
    n_basis = basis_funcs.shape[0]
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = coeffs @ basis_funcs  # shape: (n_samples, n_wavelengths)
    return spectra, coeffs

def compute_photometry(spectra, filters):
    """Integrate spectra through filter transmissions."""
    delta_lambda = np.diff(np.mean(np.vstack([filters, spectra]), axis=1))
    photometry = spectra @ filters.T * delta_lambda  # shape: (n_samples, n_filters)
    return photometry

# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def reconstruct_spectra_from_photometry(photometry, filters, basis_funcs):
    """
    Reconstruct spectra from photometric data using linear least squares.
    Assumes photometry = F @ coeffs, where F[j,i] = ∫ basis_i * filter_j.
    """
    # Build transformation matrix F (n_filters x n_basis)
    n_filters, n_wavelengths = filters.shape
    n_basis = basis_funcs.shape[0]
    F = np.zeros((n_filters, n_basis))
    for j in range(n_filters):
        for i in range(n_basis):
            F[j, i] = np.sum(filters[j] * basis_funcs[i])
    # Invert F
    F_inv = pinv(F)  # shape: (n_basis, n_filters)
    # Solve for coefficients per sample
    coeffs = F_inv @ photometry.T  # shape: (n_basis, n_samples)
    coeffs = coeffs.T              # shape: (n_samples, n_basis)
    # Reconstruct spectra
    reconstructed = coeffs @ basis_funcs  # shape: (n_samples, n_wavelengths)
    return reconstructed, coeffs

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
def main():
    # Define wavelength grid
    wavelengths = np.linspace(400, 800, 201)  # 400–800 nm, 201 points

    # Generate basis functions and filters
    basis_funcs = get_basis_functions(n_basis=5, wavelengths=wavelengths)
    filters = get_filter_transmissions(n_filters=4, wavelengths=wavelengths)

    # Generate synthetic spectra
    n_samples = 50
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis_funcs)

    # Compute photometry
    photometry = compute_photometry(spectra, filters)

    # Reconstruct spectra from photometry
    recon_spectra, recon_coeffs = reconstruct_spectra_from_photometry(photometry, filters, basis_funcs)

    # Evaluate reconstruction quality
    mse = np.mean((spectra - recon_spectra)**2)
    coeff_err = np.mean((true_coeffs - recon_coeffs)**2)
    print(f"Spectral MSE: {mse:.6f}")
    print(f"Coefficient MSE: {coeff_err:.6f}")

if __name__ == "__main__":
    main()