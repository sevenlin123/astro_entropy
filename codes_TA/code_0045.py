import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

def gaussian(x, mu, sigma):
    """One‑dimensional Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def create_basis_functions(wl, num_basis, sigma=0.02):
    """
    Generate a set of Gaussian basis functions on a common wavelength grid.
    Returns an array of shape (num_basis, len(wl)).
    """
    mus = np.linspace(0.35, 0.95, num_basis)
    return np.vstack([gaussian(wl, mu, sigma) for mu in mus])

def generate_random_coefficients(n_objects, num_basis, scale=1.0):
    """
    Draw random coefficients for each object from a normal distribution.
    Shape: (n_objects, num_basis)
    """
    return np.random.normal(scale=scale, size=(n_objects, num_basis))

def synthesize_spectra(coeffs, basis_funcs):
    """
    Linear combination of basis functions to produce spectra.
    coeffs: (n_objects, num_basis)
    basis_funcs: (num_basis, len(wl))
    Returns: (n_objects, len(wl))
    """
    return coeffs @ basis_funcs

def create_filters(wl, n_filters, width=0.08):
    """
    Create a set of Gaussian photometric filter curves.
    Returns an array of shape (n_filters, len(wl)).
    """
    centers = np.random.uniform(0.4, 0.9, size=n_filters)
    return np.vstack([gaussian(wl, c, width) for c in centers])

def compute_photometry(spectra, filters, wl):
    """
    Integrate each spectrum through each filter.
    spectra: (n_objects, len(wl))
    filters: (n_filters, len(wl))
    Returns: (n_objects, n_filters)
    """
    n_objs, n_wl = spectra.shape
    n_filt = filters.shape[0]
    phots = np.empty((n_objs, n_filt))
    for j in range(n_filt):
        phots[:, j] = simps(spectra * filters[j][None, :], wl, axis=1)
    return phots

def build_design_matrix(filters, basis_funcs, wl):
    """
    Compute the matrix A such that A[j,i] = ∫ φ_i(λ) T_j(λ) dλ
    Returns shape (n_filters, num_basis).
    """
    n_filt = filters.shape[0]
    num_basis = basis_funcs.shape[0]
    A = np.empty((n_filt, num_basis))
    for j in range(n_filt):
        for i in range(num_basis):
            A[j, i] = simps(basis_funcs[i] * filters[j], wl)
    return A

def reconstruct_spectra_from_photometry(phots, filters, basis_funcs, wl, alpha=1e-3):
    """
    Reconstruct spectra from photometric measurements using least‑squares.
    phots: (n_objects, n_filters)
    Returns reconstructed coefficients and spectra.
    """
    A = build_design_matrix(filters, basis_funcs, wl)          # (n_filt, n_basis)
    pinv_A = np.linalg.pinv(A)                                 # (n_basis, n_filt)
    coeffs_rec = phots @ pinv_A.T                              # (n_objects, n_basis)
    spectra_rec = coeffs_rec @ basis_funcs                     # (n_objects, len(wl))
    return coeffs_rec, spectra_rec

# --------------------------------------------------------------------------- #
# Example workflow
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    np.random.seed(42)

    # Wavelength grid (normalized units)
    wl = np.linspace(0.3, 1.0, 400)

    # Basis and synthetic data
    num_basis = 20
    basis_funcs = create_basis_functions(wl, num_basis)
    n_objects = 30
    coeffs_true = generate_random_coefficients(n_objects, num_basis, scale=2.0)
    spectra_true = synthesize_spectra(coeffs_true, basis_funcs)

    # Filters and photometry
    n_filters = 8
    filters = create_filters(wl, n_filters)
    phots = compute_photometry(spectra_true, filters, wl)

    # Reconstruction
    coeffs_rec, spectra_rec = reconstruct_spectra_from_photometry(
        phots, filters, basis_funcs, wl, alpha=1e-3
    )

    # Simple accuracy check
    mse = np.mean((spectra_true - spectra_rec) ** 2, axis=1)
    print("Mean reconstruction MSE over all objects:", np.mean(mse))
    print("Per‑object MSE:", mse)