import numpy as np
from sklearn.linear_model import LinearRegression

def gaussian_basis(wavelengths, n_basis, rng=None):
    """
    Create a set of Gaussian basis functions over the given wavelength grid.
    Returns the basis matrix (n_basis x N) and the parameters used.
    """
    rng = np.random.default_rng(rng)
    mu = rng.uniform(wavelengths[0], wavelengths[-1], size=n_basis)
    sigma = rng.uniform((wavelengths[-1]-wavelengths[0])/(4*n_basis),
                        (wavelengths[-1]-wavelengths[0])/(2*n_basis),
                        size=n_basis)
    basis = np.exp(-0.5*((wavelengths[:, None] - mu[None, :]) / sigma[None, :])**2)
    return basis.T, {'mu': mu, 'sigma': sigma}

def generate_synthetic_spectra(n_samples, wavelengths, n_basis, rng=None):
    """
    Generate synthetic spectra as random linear combinations of Gaussian basis functions.
    Returns the spectra matrix (n_samples x N) and the basis matrix.
    """
    basis, params = gaussian_basis(wavelengths, n_basis, rng)
    rng = np.random.default_rng(rng)
    coeffs = rng.normal(size=(n_samples, n_basis))
    spectra = coeffs @ basis
    return spectra, basis, coeffs

def generate_filters(n_filters, wavelengths, rng=None):
    """
    Generate synthetic filter transmission curves (Gaussian shaped).
    Returns a matrix (n_filters x N).
    """
    rng = np.random.default_rng(rng)
    centers = rng.uniform(wavelengths[0], wavelengths[-1], size=n_filters)
    widths = rng.uniform((wavelengths[-1]-wavelengths[0])/(10*n_filters),
                         (wavelengths[-1]-wavelengths[0])/(5*n_filters),
                         size=n_filters)
    filters = np.exp(-0.5*((wavelengths[:, None] - centers[None, :]) / widths[None, :])**2)
    return filters

def compute_photometry(spectra, filters):
    """
    Compute broadband photometry by integrating spectra over filter transmission curves.
    """
    return spectra @ filters.T

def reconstruct_spectra_from_photometry(photometry, filters, basis):
    """
    Reconstruct spectra coefficients by solving linear least squares problem.
    Returns reconstructed spectra matrix.
    """
    # Build design matrix: integral of each basis function through each filter
    A = basis @ filters.T      # shape (n_basis, n_filters)
    coeffs = []
    for p in photometry:
        # Solve A^T c = p -> c = (A A^T)^{-1} A p
        c = np.linalg.lstsq(A.T, p, rcond=None)[0]
        coeffs.append(c)
    coeffs = np.vstack(coeffs)
    return coeffs @ basis

if __name__ == "__main__":
    rng_seed = 42
    n_samples = 10
    n_basis = 5
    n_filters = 3
    wavelengths = np.linspace(400, 700, 301)  # nm
    
    # Generate synthetic data
    spectra_true, basis, true_coeffs = generate_synthetic_spectra(
        n_samples, wavelengths, n_basis, rng=rng_seed)
    
    filters = generate_filters(n_filters, wavelengths, rng=rng_seed)
    
    photometry = compute_photometry(spectra_true, filters)
    
    # Reconstruction
    spectra_rec = reconstruct_spectra_from_photometry(photometry, filters, basis)
    
    # Simple diagnostics
    for i in range(min(3, n_samples)):
        print(f"Sample {i}")
        print(f"True spectrum (first 5 values): {spectra_true[i, :5]}")
        print(f"Reconstructed spectrum (first 5 values): {spectra_rec[i, :5]}")
        print()