import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# 1. Spectral model: create a set of basis spectra (e.g. Gaussian peaks)
def generate_basis(num_bases: int, num_points: int, rng: np.random.Generator):
    """Return array of shape (num_bases, num_points)"""
    x = np.linspace(0, 1, num_points)
    bases = []
    for i in range(num_bases):
        center = rng.uniform(0.1, 0.9)
        width = rng.uniform(0.02, 0.1)
        amplitude = rng.uniform(0.5, 1.5)
        gauss = amplitude * np.exp(-0.5 * ((x - center)/width)**2)
        bases.append(gauss)
    return np.vstack(bases)

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra as linear combinations of the basis
def generate_synthetic_spectra(num_samples: int,
                               basis: np.ndarray,
                               rng: np.random.Generator):
    """Return array of shape (num_samples, num_points)"""
    num_bases, num_points = basis.shape
    coeffs = rng.normal(size=(num_samples, num_bases))
    spectra = coeffs @ basis
    # Optional: add small noise
    spectra += rng.normal(scale=0.01, size=spectra.shape)
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Create synthetic photometric filters
def generate_filters(num_filters: int, num_points: int, rng: np.random.Generator):
    """Return array of shape (num_filters, num_points)"""
    x = np.linspace(0, 1, num_points)
    filters = []
    for _ in range(num_filters):
        center = rng.uniform(0.1, 0.9)
        width = rng.uniform(0.05, 0.15)
        filt = np.exp(-0.5 * ((x - center)/width)**2)
        filt /= filt.sum()  # normalize
        filters.append(filt)
    return np.vstack(filters)

# ----------------------------------------------------------------------
# 4. Compute photometry: integrate spectrum times filter
def compute_photometry(spectra: np.ndarray, filters: np.ndarray):
    """Return array of shape (num_samples, num_filters)"""
    return spectra @ filters.T  # (samples, points) @ (filters, points).T = (samples, filters)

# ----------------------------------------------------------------------
# 5. Reconstruction: estimate coefficients via ridge regression, then rebuild spectra
def reconstruct_spectra(photometry: np.ndarray,
                        filters: np.ndarray,
                        basis: np.ndarray,
                        alpha: float = 1.0):
    """Return reconstructed spectra of shape (num_samples, num_points)"""
    # Build photometric model matrix P = filters @ basis.T
    P = filters @ basis.T  # (num_filters, num_bases)
    # Solve for coefficients using ridge
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    ridge.fit(P.T, photometry.T)  # transpose because sklearn expects samples first
    coeffs_hat = ridge.coef_.T      # shape (num_samples, num_bases)
    # Reconstruct spectra
    recon = coeffs_hat @ basis
    return recon, coeffs_hat

# ----------------------------------------------------------------------
# Demo
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Parameters
    NUM_SAMPLES = 100
    NUM_BASIS = 5
    NUM_FILTERS = 7
    NUM_POINTS = 200

    # Generate basis, spectra, filters
    basis = generate_basis(NUM_BASIS, NUM_POINTS, rng)
    spectra, true_coeffs = generate_synthetic_spectra(NUM_SAMPLES, basis, rng)
    filters = generate_filters(NUM_FILTERS, NUM_POINTS, rng)

    # Photometry
    photometry = compute_photometry(spectra, filters)

    # Reconstruction
    recon_spectra, est_coeffs = reconstruct_spectra(photometry, filters, basis, alpha=0.5)

    # Evaluate
    err = np.mean((spectra - recon_spectra)**2)
    coeff_err = np.mean((true_coeffs - est_coeffs)**2)
    print(f"Mean squared error of spectra reconstruction: {err:.6f}")
    print(f"Mean squared error of coefficient estimation: {coeff_err:.6f}")