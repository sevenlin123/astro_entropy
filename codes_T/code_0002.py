import numpy as np
from numpy.linalg import lstsq

# ------------------------------------------------------------------
# Spectral model: linear combination of basis functions
# ------------------------------------------------------------------
def generate_basis(n_wave, n_basis, seed=0):
    """
    Generate simple polynomial basis functions up to degree n_basis-1.
    Shape: (n_wave, n_basis)
    """
    rng = np.random.default_rng(seed)
    wave = np.linspace(0, 1, n_wave)
    basis = np.vstack([wave**i for i in range(n_basis)]).T
    # add small random perturbations
    basis += rng.normal(scale=0.01, size=basis.shape)
    return basis

def generate_synthetic_spectra(n_samples, basis, seed=1):
    """
    Generate spectra as random linear combinations of the basis.
    Shape: (n_samples, n_wave)
    """
    rng = np.random.default_rng(seed)
    n_basis = basis.shape[1]
    coeffs = rng.standard_normal(size=(n_samples, n_basis))
    spectra = coeffs @ basis.T
    return spectra, coeffs

# ------------------------------------------------------------------
# Filter construction and photometry generation
# ------------------------------------------------------------------
def generate_filters(n_filters, n_wave, seed=2):
    """
    Generate top‑hat filters over evenly spaced wavelength bins.
    Returns filter matrix of shape (n_filters, n_wave).
    """
    rng = np.random.default_rng(seed)
    filters = np.zeros((n_filters, n_wave))
    # Randomly choose central wavelengths and widths
    centers = rng.uniform(0, n_wave, size=n_filters)
    widths = rng.uniform(n_wave/10, n_wave/4, size=n_filters)
    for i, (c, w) in enumerate(zip(centers, widths)):
        start = int(max(c - w / 2, 0))
        end   = int(min(c + w / 2, n_wave - 1))
        filters[i, start:end] = 1.0
    return filters

def compute_photometry(spectra, filters):
    """
    Compute integrated fluxes through each filter.
    Shape: (n_samples, n_filters)
    """
    return spectra @ filters.T

# ------------------------------------------------------------------
# Reconstruction from photometry
# ------------------------------------------------------------------
def reconstruct_coefficients(photometry, filters, basis):
    """
    Estimate spectral coefficients from photometry.
    Solves: photometry ≈ coeffs @ (filters @ basis.T)
    """
    # Effective filter-weighted basis
    B = filters @ basis  # shape (n_filters, n_basis)
    # Least‑squares solution for each sample
    coeffs_est, *_ = lstsq(B.T, photometry.T, rcond=None)
    return coeffs_est.T  # shape (n_samples, n_basis)

def reconstruct_spectra(coeffs_est, basis):
    """
    Reconstruct spectra from estimated coefficients.
    """
    return coeffs_est @ basis.T

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main():
    n_wave   = 500    # number of wavelength points
    n_basis  = 5      # number of basis functions
    n_samples = 100   # number of synthetic stars
    n_filters = 3     # number of photometric bands

    # Generate basis functions
    basis = generate_basis(n_wave, n_basis)

    # Generate synthetic spectra and true coefficients
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis)

    # Generate photometric filters
    filters = generate_filters(n_filters, n_wave)

    # Compute photometry from true spectra
    photometry = compute_photometry(spectra_true, filters)

    # Reconstruct coefficients from photometry
    coeffs_est = reconstruct_coefficients(photometry, filters, basis)

    # Reconstruct spectra
    spectra_rec = reconstruct_spectra(coeffs_est, basis)

    # Example output: first sample comparison
    print("True coefficients (sample 0):", coeffs_true[0])
    print("Reconstructed coefficients (sample 0):", coeffs_est[0])
    print("\nMean absolute error in coefficients:",
          np.mean(np.abs(coeffs_true - coeffs_est)))

    # Mean squared error between true and reconstructed spectra
    mse_spec = np.mean((spectra_true - spectra_rec)**2)
    print("MSE between true and reconstructed spectra:", mse_spec)

if __name__ == "__main__":
    main()