import numpy as np
from numpy.polynomial import legendre
from sklearn.linear_model import LinearRegression

def create_wavelength_grid(start=300, stop=1000, step=1):
    """Create a wavelength grid."""
    return np.arange(start, stop + step, step)

def create_spectral_basis(wavelengths, n_components=5, seed=0):
    """Generate a set of orthogonal Legendre polynomials as a spectral basis."""
    rng = np.random.default_rng(seed)
    # normalise wavelengths to [-1, 1] for Legendre
    x = 2 * (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min()) - 1
    basis = np.column_stack([legendre.Legendre.basis(i)(x) for i in range(n_components)])
    # make them positive and normalised
    basis = np.abs(basis)
    basis /= np.linalg.norm(basis, axis=0)
    return basis

def generate_synthetic_spectra(basis, n_objects=10, seed=1):
    """Generate synthetic spectra as linear combinations of the basis."""
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(n_objects, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs

def create_filters(n_filters=3, wavelengths=None, seed=2):
    """Create simple top‑hat filters covering different wavelength ranges."""
    rng = np.random.default_rng(seed)
    if wavelengths is None:
        raise ValueError("Wavelength array must be provided.")
    filters = np.zeros((n_filters, len(wavelengths)))
    for i in range(n_filters):
        low = rng.uniform(wavelengths.min(), wavelengths.max() - 200)
        high = low + rng.uniform(100, 300)
        filters[i, (wavelengths >= low) & (wavelengths <= high)] = 1.0
    return filters

def compute_photometry(spectra, filters):
    """Compute photometric fluxes by integrating spectra through filters."""
    # spectra shape: (n_objects, n_wavelengths)
    # filters shape: (n_filters, n_wavelengths)
    # use trapezoidal integration
    fluxes = spectra @ filters.T  # approximation (no weighting)
    return fluxes

def precompute_filter_basis_matrix(filters, basis, wavelengths):
    """
    Compute the linear mapping from basis coefficients to photometry.
    Returns a matrix G such that fluxes = G @ coeffs.
    """
    n_filters, n_wavelengths = filters.shape
    n_components = basis.shape[1]
    G = np.empty((n_filters, n_components))
    for i in range(n_filters):
        for k in range(n_components):
            # integrate basis[k] * filter[i]
            integrand = basis[:, k] * filters[i]
            G[i, k] = np.trapz(integrand, wavelengths)
    return G

def reconstruct_coefficients(photometry, G):
    """Solve for basis coefficients from photometric fluxes."""
    # Use linear least squares for each object
    reg = LinearRegression(fit_intercept=False)
    reg.fit(G, photometry.T)
    coeffs_rec = reg.coef_.T
    return coeffs_rec

def reconstruct_spectra(coeffs_rec, basis):
    """Reconstruct spectra from basis coefficients."""
    return coeffs_rec @ basis.T

def main():
    wavelengths = create_wavelength_grid()
    basis = create_spectral_basis(wavelengths)
    spectra_true, coeffs_true = generate_synthetic_spectra(basis)
    filters = create_filters(n_filters=4, wavelengths=wavelengths)
    fluxes = compute_photometry(spectra_true, filters)
    G = precompute_filter_basis_matrix(filters, basis, wavelengths)
    coeffs_rec = reconstruct_coefficients(fluxes, G)
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # Simple sanity check: compare first reconstructed spectrum to true
    print("True coefficients (first object):", coeffs_true[0])
    print("Reconstructed coefficients (first object):", coeffs_rec[0])
    print("Difference (L2 norm):", np.linalg.norm(coeffs_true[0] - coeffs_rec[0]))

if __name__ == "__main__":
    main()