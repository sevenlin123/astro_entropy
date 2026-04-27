import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Spectral model utilities
# ------------------------------------------------------------------

def gaussian_basis(wavelength, mu, sigma):
    """Return a single Gaussian basis function evaluated on wavelength grid."""
    return np.exp(-0.5 * ((wavelength - mu) / sigma) ** 2)

def build_basis_functions(n_components, wavelength_grid, sigma=0.05):
    """
    Build a set of Gaussian basis functions.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components.
    wavelength_grid : ndarray
        Array of wavelength points.
    sigma : float, optional
        Standard deviation of Gaussians.

    Returns
    -------
    basis : ndarray of shape (n_components, len(wavelength_grid))
        Each row is a basis function evaluated on the grid.
    """
    mus = np.linspace(wavelength_grid.min(), wavelength_grid.max(),
                      n_components)
    basis = np.vstack([gaussian_basis(wavelength_grid, mu, sigma)
                       for mu in mus])
    return basis

# ------------------------------------------------------------------
# Synthetic spectrum generation
# ------------------------------------------------------------------

def generate_synthetic_spectra(n_spectra, basis_functions):
    """
    Generate synthetic spectra as linear combinations of basis functions.

    Parameters
    ----------
    n_spectra : int
        Number of synthetic spectra to generate.
    basis_functions : ndarray of shape (n_components, n_wavelength)
        Basis functions.

    Returns
    -------
    spectra : ndarray of shape (n_spectra, n_wavelength)
        Generated spectra.
    true_coeffs : ndarray of shape (n_spectra, n_components)
        Coefficients used to build the spectra.
    """
    n_components, n_wavelength = basis_functions.shape
    rng = np.random.default_rng()
    true_coeffs = rng.normal(loc=1.0, scale=0.5,
                             size=(n_spectra, n_components))
    spectra = true_coeffs @ basis_functions
    return spectra, true_coeffs

# ------------------------------------------------------------------
# Filter generation and photometry
# ------------------------------------------------------------------

def build_filter_transmissions(n_filters, wavelength_grid, sigma=0.1):
    """
    Create synthetic filter transmission curves.

    Parameters
    ----------
    n_filters : int
        Number of filters.
    wavelength_grid : ndarray
        Wavelength grid.
    sigma : float, optional
        Width of filter Gaussians.

    Returns
    -------
    filters : ndarray of shape (n_filters, len(wavelength_grid))
        Filter transmission curves.
    """
    mus = np.linspace(wavelength_grid.min() + 0.1 * (wavelength_grid.max()-wavelength_grid.min()),
                      wavelength_grid.max() - 0.1 * (wavelength_grid.max()-wavelength_grid.min()),
                      n_filters)
    filters = np.vstack([gaussian_basis(wavelength_grid, mu, sigma) for mu in mus])
    # Normalize to max 1
    filters /= filters.max(axis=1, keepdims=True)
    return filters

def compute_photometry(spectra, filters):
    """
    Integrate spectra over filter transmissions to obtain photometric fluxes.

    Parameters
    ----------
    spectra : ndarray of shape (n_spectra, n_wavelength)
    filters : ndarray of shape (n_filters, n_wavelength)

    Returns
    -------
    photometry : ndarray of shape (n_spectra, n_filters)
    """
    # Simple Riemann sum integration
    return spectra @ filters.T

# ------------------------------------------------------------------
# Spectrum reconstruction
# ------------------------------------------------------------------

def reconstruct_spectra(photometry, filters, basis_functions):
    """
    Reconstruct spectra from photometry using linear regression.

    Parameters
    ----------
    photometry : ndarray of shape (n_spectra, n_filters)
    filters : ndarray of shape (n_filters, n_wavelength)
    basis_functions : ndarray of shape (n_components, n_wavelength)

    Returns
    -------
    reconstructed_spectra : ndarray of shape (n_spectra, n_wavelength)
    estimated_coeffs : ndarray of shape (n_spectra, n_components)
    """
    # Compute basis integrals through filters
    # For each basis function, compute its integrated response in each filter
    # response_matrix shape: (n_filters, n_components)
    response_matrix = basis_functions @ filters.T  # (n_components, n_filters).T -> (n_filters, n_components)
    # Fit linear regression: photometry = response_matrix * coeffs
    reg = LinearRegression(fit_intercept=False)
    reg.fit(response_matrix, photometry.T)  # Transpose to shape (n_filters, n_spectra)
    estimated_coeffs = reg.coef_.T          # shape (n_spectra, n_components)
    reconstructed_spectra = estimated_coeffs @ basis_functions
    return reconstructed_spectra, estimated_coeffs

# ------------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------------

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define wavelength grid
    wavelength_grid = np.linspace(0.4, 0.8, 400)  # e.g., microns

    # Build basis functions
    n_components = 5
    basis_funcs = build_basis_functions(n_components, wavelength_grid, sigma=0.02)

    # Generate synthetic spectra
    n_spectra = 10
    spectra, true_coeffs = generate_synthetic_spectra(n_spectra, basis_funcs)

    # Build filter transmissions
    n_filters = 3
    filters = build_filter_transmissions(n_filters, wavelength_grid, sigma=0.05)

    # Compute photometry
    photometry = compute_photometry(spectra, filters)

    # Reconstruct spectra
    recon_spectra, est_coeffs = reconstruct_spectra(photometry, filters, basis_funcs)

    # Simple evaluation: mean absolute error
    mae = np.mean(np.abs(recon_spectra - spectra))
    print(f"Mean absolute reconstruction error: {mae:.4f}")

    # Print sample spectra comparison
    idx = 0
    print("\nTrue spectrum (sample):")
    print(spectra[idx])
    print("\nReconstructed spectrum (sample):")
    print(recon_spectra[idx])

if __name__ == "__main__":
    main()