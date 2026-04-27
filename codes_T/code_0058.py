import numpy as np
from scipy.linalg import lstsq

# --------------------------------------------------------------------
# 1. Spectral model ---------------------------------------------------
# --------------------------------------------------------------------
def generate_basis(n_wavelengths: int, n_basis: int, seed: int | None = None):
    """
    Create an orthonormal set of basis spectra (e.g., sinusoids).
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(0, 1, n_wavelengths)
    basis = np.array([np.sin((i + 1) * np.pi * wavelengths) for i in range(n_basis)])
    # Normalize each basis vector
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    return basis.T  # shape (n_wavelengths, n_basis)

# --------------------------------------------------------------------
# 2. Generate synthetic spectra ---------------------------------------
# --------------------------------------------------------------------
def synthesize_spectra(n_samples: int, basis, seed: int | None = None):
    """
    Generate spectra as random linear combinations of the given basis.
    """
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(-1, 1, size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T  # shape (n_samples, n_wavelengths)
    return spectra, coeffs

# --------------------------------------------------------------------
# 3. Photometric data from spectra ------------------------------------
# --------------------------------------------------------------------
def generate_random_filters(n_filters: int, n_wavelengths: int,
                            width_range=(0.05, 0.15), seed: int | None = None):
    """
    Create random bandpass filters as Gaussian-shaped transmission curves.
    """
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(0.1, 0.9)
        width = rng.uniform(*width_range)
        x = np.linspace(0, 1, n_wavelengths)
        trans = np.exp(-0.5 * ((x - center) / width) ** 2)
        trans /= trans.max()  # normalize to peak 1
        filters.append(trans)
    return np.array(filters)  # shape (n_filters, n_wavelengths)

def compute_photometry(spectra: np.ndarray, filters: np.ndarray):
    """
    Integrate spectra over each filter to obtain photometric fluxes.
    """
    # Assume flat wavelength spacing; integral ~ sum * delta_lambda
    return spectra @ filters.T  # shape (n_samples, n_filters)

# --------------------------------------------------------------------
# 4. Reconstruction ---------------------------------------------------
# --------------------------------------------------------------------
def reconstruct_spectra(photometry: np.ndarray,
                        filters: np.ndarray,
                        basis: np.ndarray,
                        reg_lambda: float = 1e-6):
    """
    Reconstruct spectra from photometry via least‑squares estimation
    of the linear coefficients that map filters to basis projections.
    """
    # Build design matrix that maps coefficients to photometry
    # photometry = coeffs @ (basis^T @ filters^T)
    design = basis.T @ filters.T   # shape (n_basis, n_filters)
    # Solve for coefficients per sample
    coeffs, *_ = lstsq(design.T, photometry.T, lapack_driver='gelsy')
    coeffs = coeffs.T  # shape (n_samples, n_basis)
    # Reconstruct spectra
    reconstructed = coeffs @ basis.T
    return reconstructed

# --------------------------------------------------------------------
# Example usage --------------------------------------------------------
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    n_wavelengths = 200
    n_basis = 5
    n_samples = 50
    n_filters = 10
    seed = 42

    # 1. Basis
    basis = generate_basis(n_wavelengths, n_basis, seed=seed)

    # 2. Synthetic spectra
    spectra, true_coeffs = synthesize_spectra(n_samples, basis, seed=seed)

    # 3. Filters & photometry
    filters = generate_random_filters(n_filters, n_wavelengths,
                                      width_range=(0.04, 0.12), seed=seed+1)
    photometry = compute_photometry(spectra, filters)

    # 4. Reconstruction
    reconstructed = reconstruct_spectra(photometry, filters, basis)

    # Evaluate reconstruction error
    error = np.linalg.norm(reconstructed - spectra) / np.linalg.norm(spectra)
    print(f"Reconstruction relative RMS error: {error:.4f}")