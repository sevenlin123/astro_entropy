import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# --- 1. Spectral model -------------------------------------------------------
def wavelength_grid(n=500, lam_min=300, lam_max=1100):
    """Generate a wavelength grid in nm."""
    return np.linspace(lam_min, lam_max, n)

def gaussian_basis(lam, centers, widths):
    """Return a set of Gaussian basis functions evaluated on `lam`."""
    basis = []
    for c, w in zip(centers, widths):
        basis.append(np.exp(-0.5 * ((lam - c) / w)**2))
    return np.vstack(basis)  # shape (n_bases, n_lam)

def spectral_flux(coeffs, basis):
    """Linear combination of basis functions."""
    return basis.T @ coeffs  # shape (n_lam,)

# --- 2. Synthetic spectra ----------------------------------------------------
def generate_random_coeffs(n_spectra, n_bases, seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_spectra, n_bases))

def synthetic_spectra(lam, basis, coeffs):
    """Compute all spectra for given coefficients."""
    return basis.T @ coeffs.T  # shape (n_lam, n_spectra)

# --- 3. Photometric data -----------------------------------------------------
def filter_response(lam, center, width):
    """Simple Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((lam - center) / width)**2)

def photometric_matrix(lam, basis, filters):
    """
    Construct the matrix that maps basis coefficients to photometric band fluxes.
    filters : list of tuples (center, width)
    """
    n_bands = len(filters)
    n_bases = basis.shape[0]
    P = np.zeros((n_bands, n_bases))
    for i, (c, w) in enumerate(filters):
        filt = filter_response(lam, c, w)
        for j in range(n_bases):
            integrand = filt * basis[j]
            P[i, j] = simps(integrand, lam)
    return P

def generate_photometry(P, coeffs):
    """Compute photometric measurements from coefficient matrix."""
    return P @ coeffs.T  # shape (n_bands, n_spectra)

# --- 4. Reconstruction ------------------------------------------------------
def reconstruct_coeffs(P, photometry, alpha=0.01):
    """Recover basis coefficients from photometry using ridge regression."""
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(P, photometry.T)
    return ridge.coef_.T  # shape (n_bases, n_spectra)

def reconstruct_spectra(lam, basis, coeffs):
    """Reconstruct spectra from recovered coefficients."""
    return basis.T @ coeffs.T  # shape (n_lam, n_spectra)

# --- Main demonstration ------------------------------------------------------
def main():
    # Grid and basis
    lam = wavelength_grid()
    n_bases = 5
    centers = np.linspace(350, 950, n_bases)
    widths = np.full(n_bases, 50.)
    basis = gaussian_basis(lam, centers, widths)

    # Filters (e.g., U, B, V, R, I)
    filters = [(365, 30), (445, 30), (551, 30), (658, 30), (806, 30)]

    # Generate synthetic data
    n_spectra = 10
    coeffs_true = generate_random_coeffs(n_spectra, n_bases, seed=42)
    spectra_true = synthetic_spectra(lam, basis, coeffs_true)

    # Photometric measurements
    P = photometric_matrix(lam, basis, filters)
    photometry = generate_photometry(P, coeffs_true)

    # Reconstruction
    coeffs_rec = reconstruct_coeffs(P, photometry, alpha=0.05)
    spectra_rec = reconstruct_spectra(lam, basis, coeffs_rec)

    # Simple comparison
    print("True vs reconstructed spectra (first star):")
    print("wavelength (nm)\ttrue_flux\trec_flux")
    for l, t, r in zip(lam[::100], spectra_true[::100, 0], spectra_rec[::100, 0]):
        print(f"{l:.1f}\t{t:.4f}\t{r:.4f}")

if __name__ == "__main__":
    main()