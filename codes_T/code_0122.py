import numpy as np
from scipy.integrate import simps
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# 1. Spectral model
# ------------------------------------------------------------------
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

def create_basis_functions(n_components, wavelength_grid):
    """Create a set of Gaussian basis functions."""
    np.random.seed(0)
    mus = np.linspace(wavelength_grid.min(), wavelength_grid.max(), n_components)
    sigmas = np.full(n_components, (wavelength_grid.max()-wavelength_grid.min())/(4*n_components))
    basis = np.vstack([gaussian(wavelength_grid, mu, sigma) for mu, sigma in zip(mus, sigmas)])
    return basis  # shape (n_components, len(grid))

# ------------------------------------------------------------------
# 2. Synthetic spectra
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis, weight_bounds=(0.1, 1.0)):
    """Generate spectra as linear combinations of basis functions."""
    rng = np.random.default_rng(1)
    weights = rng.uniform(weight_bounds[0], weight_bounds[1], size=(n_samples, basis.shape[0]))
    spectra = weights @ basis  # shape (n_samples, len(grid))
    return spectra, weights

# ------------------------------------------------------------------
# 3. Photometric data
# ------------------------------------------------------------------
def create_filter_curves(n_filters, wavelength_grid):
    """Generate simple Gaussian filter curves."""
    np.random.seed(2)
    centers = np.linspace(wavelength_grid.min()+0.2*(wavelength_grid.max()-wavelength_grid.min()),
                          wavelength_grid.max()-0.2*(wavelength_grid.max()-wavelength_grid.min()),
                          n_filters)
    widths = np.full(n_filters, (wavelength_grid.max()-wavelength_grid.min())/(3*n_filters))
    filters = np.vstack([gaussian(wavelength_grid, c, w) for c, w in zip(centers, widths)])
    # normalize so that integral over wavelength = 1
    filters /= simps(filters, wavelength_grid, axis=1, keepdims=True)
    return filters  # shape (n_filters, len(grid))

def compute_photometry(spectra, filters, wavelength_grid):
    """Integrate spectra through filters to get photometric fluxes."""
    # spectra: (n_samples, len(grid))
    # filters: (n_filters, len(grid))
    integrals = np.array([
        simps(spectra * filt, wavelength_grid, axis=1)
        for filt in filters
    ])  # shape (n_filters, n_samples)
    return integrals.T  # shape (n_samples, n_filters)

# ------------------------------------------------------------------
# 4. Reconstruction
# ------------------------------------------------------------------
def reconstruct_spectra(photon_fluxes, filters, basis, wavelength_grid):
    """
    Recover spectral coefficients from photometric fluxes by solving a linear system.
    The relation between coefficients and photometric fluxes is:
        F = (B @ C.T) * T   =>  F = (T @ B).T @ C
    where B are basis, T are filter curves.
    """
    # Compute effective mapping matrix M = T @ B (filters x basis)
    M = filters @ basis.T  # shape (n_filters, n_components)
    # Solve for coefficients using non-negative least squares for each sample
    coeffs = np.zeros((photon_fluxes.shape[0], basis.shape[0]))
    for i, f in enumerate(photon_fluxes):
        coeffs[i], _ = nnls(M, f)
    # Reconstruct spectra
    reconstructed = coeffs @ basis
    return reconstructed, coeffs

# ------------------------------------------------------------------
# Example workflow
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Define wavelength grid
    wl_min, wl_max = 300.0, 800.0  # nm
    nw = 500
    wl_grid = np.linspace(wl_min, wl_max, nw)

    # Create spectral basis
    n_basis = 5
    basis = create_basis_functions(n_basis, wl_grid)

    # Generate synthetic spectra
    n_samples = 200
    spectra, true_weights = generate_synthetic_spectra(n_samples, basis)

    # Create filter curves
    n_filters = 3
    filters = create_filter_curves(n_filters, wl_grid)

    # Compute photometric fluxes
    photometry = compute_photometry(spectra, filters, wl_grid)

    # Reconstruct spectra
    recon_spectra, recon_weights = reconstruct_spectra(photometry, filters, basis, wl_grid)

    # Evaluate reconstruction error
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared reconstruction error: {mse:.6f}")

    # Compare true vs recovered weights for first sample
    print("\nTrue weights for first sample:", true_weights[0])
    print("Recovered weights for first sample:", recon_weights[0])