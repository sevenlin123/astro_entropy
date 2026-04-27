import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------
# Spectral basis
# ----------------------------------------------------------------------
def generate_basis(n_wavelengths, n_bases, rng=None):
    """Generate random positive spectral basis functions."""
    rng = np.random.default_rng(rng)
    return rng.uniform(low=0.1, high=1.0, size=(n_bases, n_wavelengths))


# ----------------------------------------------------------------------
# Synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(basis, n_objects, noise_std=0.01, rng=None):
    """
    Create synthetic spectra as random linear combinations of the basis
    with added Gaussian noise.
    """
    rng = np.random.default_rng(rng)
    coeffs = rng.normal(size=(n_objects, basis.shape[0]))
    spectra = coeffs @ basis
    noise = rng.normal(scale=noise_std, size=spectra.shape)
    return spectra + noise, coeffs


# ----------------------------------------------------------------------
# Filter definitions
# ----------------------------------------------------------------------
def create_filters(n_filters, n_wavelengths, rng=None):
    """
    Generate simple Gaussian filters with different central wavelengths
    and widths.
    """
    rng = np.random.default_rng(rng)
    lam = np.linspace(0.4, 1.0, n_wavelengths)  # arbitrary wavelength grid
    filters = np.zeros((n_filters, n_wavelengths))
    centers = rng.uniform(0.45, 0.95, size=n_filters)
    widths = rng.uniform(0.05, 0.15, size=n_filters)
    for i, (c, w) in enumerate(zip(centers, widths)):
        filters[i] = norm.pdf(lam, loc=c, scale=w)
        filters[i] /= filters[i].sum()  # normalize
    return filters


# ----------------------------------------------------------------------
# Photometric data
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """
    Compute photometric fluxes by integrating spectra with filter curves.
    """
    return spectra @ filters.T  # (n_obj, n_filter)


# ----------------------------------------------------------------------
# Spectrum reconstruction
# ----------------------------------------------------------------------
def reconstruct_spectra_from_photometry(photometry, filters, basis, eps=1e-12):
    """
    Recover linear coefficients by solving the linear system
    Y = C @ M, where M = basis @ filters.T .
    Returns reconstructed spectra.
    """
    # Design matrix M (n_filter, n_basis)
    M = filters @ basis.T   # shape (n_filter, n_basis)

    # Pseudo-inverse of M.T (shape n_basis, n_filter)
    M_t_inv = np.linalg.pinv(M.T, rcond=eps)

    # Estimate coefficients C (n_obj, n_basis)
    coeffs_est = photometry @ M_t_inv

    # Reconstruct spectra
    return coeffs_est @ basis


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    N_WL = 100      # number of wavelength points
    N_BAS = 5       # number of basis spectra
    N_OBJ = 20      # number of synthetic objects
    N_FILT = 3      # number of filters

    # Step 1: create basis
    basis = generate_basis(N_WL, N_BAS, rng=42)

    # Step 2: generate synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(
        basis, N_OBJ, noise_std=0.02, rng=123
    )

    # Step 3: create filters
    filters = create_filters(N_FILT, N_WL, rng=99)

    # Step 4: compute photometric data
    photometry = compute_photometry(spectra, filters)

    # Step 5: reconstruct spectra from photometry
    recon_spectra = reconstruct_spectra_from_photometry(
        photometry, filters, basis, eps=1e-12
    )

    # Evaluate reconstruction quality
    rmse = np.sqrt(((spectra - recon_spectra) ** 2).mean(axis=1))
    print("RMSE per object:", rmse)
    print("Average RMSE:", rmse.mean())