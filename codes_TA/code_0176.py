import numpy as np
from scipy.stats import norm

# ---------------------------------------------
# Spectral model and helper functions
# ---------------------------------------------
def gaussian(wl, center, width):
    """Simple Gaussian function."""
    return np.exp(-0.5 * ((wl - center) / width)**2)

def generate_basis(wl, n_basis, rng):
    """Generate a set of Gaussian basis functions."""
    centers = np.linspace(wl[0], wl[-1], n_basis)
    widths  = (wl[-1] - wl[0]) / (2 * n_basis)
    basis   = [gaussian(wl, c, widths) for c in centers]
    return np.vstack(basis)  # shape: (n_basis, len(wl))

def generate_filters(wl, n_filters, rng):
    """Generate Gaussian photometric filters."""
    centers = np.linspace(wl[0] + (wl[-1]-wl[0])/6,
                         wl[-1] - (wl[-1]-wl[0])/6,
                         n_filters)
    widths  = (wl[-1] - wl[0]) / (4 * n_filters)
    filters = [gaussian(wl, c, widths) for c in centers]
    return np.vstack(filters)  # shape: (n_filters, len(wl))

def normalize(v):
    return v / np.linalg.norm(v)

# ---------------------------------------------
# Synthetic data generation
# ---------------------------------------------
def generate_synthetic_spectra(n_spectra, basis, rng):
    """Generate random linear combinations of basis functions."""
    n_basis = basis.shape[0]
    coeffs  = rng.standard_normal((n_spectra, n_basis))
    coeffs  = np.apply_along_axis(normalize, 1, coeffs)
    spectra = coeffs @ basis  # shape: (n_spectra, len(wl))
    return spectra, coeffs

def compute_photometry(spectra, filters, wl):
    """Integrate spectra over filter bandpasses."""
    delta_wl = wl[1] - wl[0]
    # Precompute integrals: (filters * spectra).sum(axis=1)
    phot = spectra @ (filters.T * delta_wl)
    return phot  # shape: (n_spectra, n_filters)

# ---------------------------------------------
# Reconstruction
# ---------------------------------------------
def reconstruct_spectra(photometry, filters, basis, wl):
    """
    Reconstruct spectra by solving linear least‑squares for basis
    coefficients from photometry.
    """
    delta_wl = wl[1] - wl[0]
    # Build design matrix: M_{ij} = ∫ filter_j * basis_i dλ
    M = (filters @ basis.T) * delta_wl  # shape: (n_filters, n_basis)
    # Solve for each spectrum
    coeffs_rec = np.linalg.lstsq(M, photometry.T, rcond=None)[0].T
    spectra_rec = coeffs_rec @ basis  # shape: (n_spectra, len(wl))
    return spectra_rec, coeffs_rec

# ---------------------------------------------
# Demo
# ---------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # Wavelength grid (nm)
    wl = np.linspace(400, 700, 1000)

    # Generate basis functions and filters
    n_basis = 10
    n_filters = 12
    basis   = generate_basis(wl, n_basis, rng)
    filters = generate_filters(wl, n_filters, rng)

    # Synthetic spectra
    n_spectra = 5
    spectra, coeffs_true = generate_synthetic_spectra(n_spectra, basis, rng)

    # Photometry
    phot = compute_photometry(spectra, filters, wl)

    # Reconstruction
    spectra_rec, coeffs_rec = reconstruct_spectra(phot, filters, basis, wl)

    # Evaluation
    rmse = np.sqrt(np.mean((spectra - spectra_rec)**2, axis=1))
    for i, e in enumerate(rmse, 1):
        print(f"Spectrum {i}: RMSE = {e:.3f}")

if __name__ == "__main__":
    main()