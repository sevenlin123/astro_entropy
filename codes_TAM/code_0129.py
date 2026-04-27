import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
# Basic utilities
# --------------------------------------------------------------------------- #

def create_wavelength_grid(start=300, stop=1000, npts=200):
    """Create a linear wavelength grid (nm)."""
    return np.linspace(start, stop, npts)


def gaussian(x, mu, sigma):
    """One‑dimensional Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# --------------------------------------------------------------------------- #
# Spectral model
# --------------------------------------------------------------------------- #

def generate_basis_functions(lam, n_basis=5, seed=None):
    """Generate random Gaussian basis functions."""
    rng = np.random.default_rng(seed)
    mus = rng.uniform(lam.min(), lam.max(), n_basis)
    sigmas = rng.uniform((lam.max() - lam.min()) / 10,
                         (lam.max() - lam.min()) / 4,
                         n_basis)
    basis = np.vstack([gaussian(lam, mu, sigma) for mu, sigma in zip(mus, sigmas)])
    return basis  # shape (n_basis, len(lam))


def synthesize_spectrum(coeffs, basis):
    """Linear combination of basis functions."""
    return coeffs @ basis


# --------------------------------------------------------------------------- #
# Photometric simulation
# --------------------------------------------------------------------------- #

def generate_filters(lam, n_filters=5, seed=None):
    """Create simple top‑hat filter response matrices."""
    rng = np.random.default_rng(seed)
    fwhms = rng.uniform((lam.max() - lam.min()) / 12,
                        (lam.max() - lam.min()) / 6,
                        n_filters)
    edges = np.cumsum(fwhms) + lam.min()
    filters = np.zeros((n_filters, len(lam)))
    for i, edge in enumerate(edges[:n_filters]):
        left = edge - fwhms[i] / 2
        right = edge + fwhms[i] / 2
        mask = (lam >= left) & (lam <= right)
        filters[i, mask] = 1.0
    return filters  # shape (n_filters, len(lam))


def simulate_photometry(spectrum, filters):
    """Integrate spectrum through each filter."""
    return filters @ spectrum  # shape (n_filters,)


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #

def build_filter_basis_matrix(filters, basis):
    """
    Compute the mapping from basis coefficients to photometric measurements:
    photometry = A @ coeffs
    """
    n_filters, n_lam = filters.shape
    n_basis = basis.shape[0]
    A = np.zeros((n_filters, n_basis))
    for m in range(n_filters):
        for b in range(n_basis):
            A[m, b] = np.trapz(filters[m] * basis[b], axis=-1)
    return A


def reconstruct_spectrum(photometry, filters, basis, alpha=1.0):
    """
    Reconstruct spectrum from photometric data using ridge regression.
    Returns the full wavelength‑dependent spectrum.
    """
    A = build_filter_basis_matrix(filters, basis)          # (M, B)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, photometry)
    coeffs = ridge.coef_                                   # (B,)
    return synthesize_spectrum(coeffs, basis)              # (N,)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Grid and models
    lam = create_wavelength_grid()
    basis = generate_basis_functions(lam, n_basis=8, seed=42)
    filters = generate_filters(lam, n_filters=6, seed=24)

    # Synthetic data
    n_samples = 10
    true_specs = []
    phot_data = []

    rng = np.random.default_rng(123)
    for _ in range(n_samples):
        coeffs = rng.normal(loc=0.0, scale=1.0, size=basis.shape[0])
        spec = synthesize_spectrum(coeffs, basis)
        true_specs.append(spec)
        phot = simulate_photometry(spec, filters)
        phot_data.append(phot)

    # Reconstruction
    recon_specs = [reconstruct_spectrum(p, filters, basis, alpha=0.01)
                   for p in phot_data]

    # Evaluate
    for i, (orig, recon) in enumerate(zip(true_specs, recon_specs)):
        corr = np.corrcoef(orig, recon)[0, 1]
        print(f"Sample {i+1:02d}: correlation = {corr:.3f}")