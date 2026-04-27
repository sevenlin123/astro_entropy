import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def create_spectral_basis(wl, n_basis):
    """Return an array of shape (n_basis, len(wl)) containing Gaussian basis functions."""
    centers = np.linspace(wl[0], wl[-1], n_basis)
    widths = (wl[-1] - wl[0]) / (2 * n_basis)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wl - c) / widths) ** 2)
        basis.append(g)
    return np.array(basis)

def generate_synthetic_spectra(n_samples, basis, rng=None):
    """Generate random linear combinations of the basis functions."""
    rng = rng or np.random.default_rng()
    coeffs = rng.normal(size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis
    return spectra, coeffs

# ---------- Filter model ----------
def create_filter_set(n_filters, wl):
    """Return an array of shape (n_filters, len(wl)) containing Gaussian filters."""
    centers = np.linspace(wl[0], wl[-1], n_filters)
    widths = (wl[-1] - wl[0]) / (4 * n_filters)
    filters = []
    for c in centers:
        f = np.exp(-0.5 * ((wl - c) / widths) ** 2)
        filters.append(f)
    return np.array(filters)

def compute_photometry(spectra, filters, wl):
    """Integrate spectra through filters to obtain photometric fluxes."""
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    phot = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        filt = filters[i]
        # Normalize filter to unit area
        norm = simps(filt, wl)
        for j in range(n_samples):
            phot[j, i] = simps(spectra[j] * filt, wl) / norm
    return phot

# ---------- Reconstruction ----------
def build_design_matrix(filters, basis, wl):
    """Construct matrix M where M[i, j] = ∫ basis_j * filter_i / ∫ filter_i."""
    n_filters, n_basis = filters.shape[0], basis.shape[0]
    M = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        filt = filters[i]
        norm = simps(filt, wl)
        for j in range(n_basis):
            M[i, j] = simps(basis[j] * filt, wl) / norm
    return M

def reconstruct_spectrum_from_photometry(phot, M, basis):
    """Given photometry vector, reconstruct spectrum via least‑squares."""
    reg = LinearRegression(fit_intercept=False).fit(M, phot)
    coeffs = reg.coef_
    reconstructed = coeffs @ basis
    return reconstructed, coeffs

# ---------- Main demonstration ----------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Wavelength grid
    wl = np.linspace(3000, 10000, 2000)  # Angstroms

    # Basis and filters
    n_basis = 10
    n_filters = 5
    basis = create_spectral_basis(wl, n_basis)
    filters = create_filter_set(n_filters, wl)

    # Generate synthetic spectra
    n_samples = 100
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis, rng)

    # Compute photometric observations
    phot = compute_photometry(spectra, filters, wl)

    # Build design matrix once
    M = build_design_matrix(filters, basis, wl)

    # Reconstruct first spectrum
    idx = 0
    rec_spec, rec_coeffs = reconstruct_spectrum_from_photometry(phot[idx], M, basis)

    # Evaluate reconstruction error
    mse = np.mean((rec_spec - spectra[idx])**2)
    print(f"Mean squared reconstruction error (first sample): {mse:.4e}")