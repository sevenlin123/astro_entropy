import numpy as np
from scipy.integrate import simps

# ------------------------------------------------------------------
# 1. Spectral model – basis functions
# ------------------------------------------------------------------
def gaussian(x, mu, sigma):
    """Simple Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def build_basis(wave, n_basis=5):
    """Return basis matrix B (wavepoints × n_basis)."""
    basis = []
    centers = np.linspace(450, 650, n_basis)          # µ in nm
    sigmas  = np.full(n_basis, 30.0)                  # σ in nm
    for mu, sigma in zip(centers, sigmas):
        basis.append(gaussian(wave, mu, sigma))
    return np.column_stack(basis)                    # shape (len(wave), n_basis)

# ------------------------------------------------------------------
# 2. Synthetic spectra generation
# ------------------------------------------------------------------
def generate_spectra(n_spectra, basis):
    """
    Random linear combinations of basis functions.
    Returns array of shape (n_spectra, n_wave).
    """
    coeffs = np.random.randn(n_spectra, basis.shape[1])  # random coefficients
    return coeffs @ basis.T                               # (n_spectra, n_wave)

# ------------------------------------------------------------------
# 3. Filters (photometric system)
# ------------------------------------------------------------------
def build_filters(wave, n_filter=3):
    """Return filter matrix F (n_filter × n_wave)."""
    filters = []
    centers = np.array([350, 500, 650])  # nm
    sigmas  = np.full(n_filter, 50.0)    # nm
    for mu, sigma in zip(centers, sigmas):
        filters.append(gaussian(wave, mu, sigma))
    return np.vstack(filters)            # (n_filter, n_wave)

def compute_photometry(spectra, filters):
    """
    Integrate spectra through filters.
    spectra: (n_spectra, n_wave)
    filters: (n_filter, n_wave)
    Returns photometry array (n_spectra, n_filter)
    """
    # Apply Simpson integration along wave axis
    phots = np.zeros((spectra.shape[0], filters.shape[0]))
    for i in range(filters.shape[0]):
        phots[:, i] = simps(spectra * filters[i, :], x=None, dx=(wave[1]-wave[0]), axis=1)
    return phots

# ------------------------------------------------------------------
# 4. Spectrum reconstruction from photometry
# ------------------------------------------------------------------
def reconstruct_from_photometry(photometry, basis, filters):
    """
    Linear least‑squares inversion:
        p = M * c   where  M = filters @ basis
    Solve for c and then reconstruct spectra.
    """
    # Build linear operator M
    M = filters @ basis                      # shape (n_filter, n_basis)
    # Solve for coefficients using pseudoinverse
    coeffs_hat = np.linalg.lstsq(M, photometry.T, rcond=None)[0].T  # (n_spectra, n_basis)
    # Reconstruct spectra
    spectra_hat = coeffs_hat @ basis.T       # (n_spectra, n_wave)
    return spectra_hat

# ------------------------------------------------------------------
# 5. Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wave = np.linspace(400, 700, 300)         # nm

    # Basis
    basis = build_basis(wave, n_basis=5)      # (300,5)

    # Filters
    filters = build_filters(wave, n_filter=3) # (3,300)

    # Synthetic spectra
    n_obj = 10
    true_spectra = generate_spectra(n_obj, basis)

    # Photometry
    photometry = compute_photometry(true_spectra, filters)

    # Reconstruction
    recon_spectra = reconstruct_from_photometry(photometry, basis, filters)

    # Error metric
    err = np.linalg.norm(true_spectra - recon_spectra) / np.linalg.norm(true_spectra)
    print(f"Relative reconstruction error: {err:.3%}")