import numpy as np
from scipy.special import legendre
from sklearn.linear_model import Ridge

# ------------------ Model definition ------------------
def build_basis(wl, n_basis):
    """Construct a basis matrix using normalized Legendre polynomials."""
    wl_norm = (wl - wl.mean()) / (wl.ptp() / 2.0)  # scale to [-1,1]
    basis = np.column_stack([legendre(k)(wl_norm) for k in range(n_basis)])
    return basis  # shape (len(wl), n_basis)

# ------------------ Synthetic data generation ------------------
def generate_synthetic_spectra(n_samples, n_basis, wl, rng=None):
    """Generate spectra as linear combinations of basis functions."""
    rng = np.random.default_rng(rng)
    basis = build_basis(wl, n_basis)
    coeffs = rng.normal(size=(n_samples, n_basis))
    spectra = coeffs @ basis.T          # shape (n_samples, len(wl))
    return spectra, coeffs

def generate_filters(n_filters, wl):
    """Create simple top‑hat filter responses."""
    rng = np.random.default_rng()
    filters = []
    wl_min, wl_max = wl.min(), wl.max()
    for _ in range(n_filters):
        a = rng.uniform(wl_min, wl_max)
        b = rng.uniform(a, wl_max)
        filt = np.where((wl >= a) & (wl <= b), 1.0, 0.0)
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, len(wl))

def compute_photometry(spectra, filters, wl):
    """Integrate spectra through filters to get fluxes."""
    dw = np.diff(wl, prepend=wl[0])  # assume uniform spacing where possible
    phot = spectra @ (filters.T * dw)  # shape (n_samples, n_filters)
    return phot

# ------------------ Reconstruction framework ------------------
def build_photometry_matrix(basis, filters, wl):
    """
    Precompute matrix that maps spectral coefficients to photometric fluxes.
    """
    dw = np.diff(wl, prepend=wl[0])
    # Integrate each basis function over each filter
    P = np.zeros((filters.shape[0], basis.shape[1]))
    for f_idx, filt in enumerate(filters):
        for b_idx in range(basis.shape[1]):
            integrand = basis[:, b_idx] * filt
            P[f_idx, b_idx] = np.sum(integrand * dw)
    return P

def reconstruct_coefficients(phot, P, alpha=1.0):
    """Solve for spectral coefficients using ridge regression."""
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(P, phot.T)  # P: (n_filters, n_basis)
    coeff_est = reg.predict(phot.T).T   # shape (n_samples, n_basis)
    return coeff_est

def reconstruct_spectra(coeff_est, basis):
    """Reconstruct spectra from estimated coefficients."""
    return coeff_est @ basis.T  # shape (n_samples, len(wl))

# ------------------ Demo ------------------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(400, 800, 500)  # nm

    # Parameters
    n_samples = 200
    n_basis   = 7
    n_filters = 4

    # Generate synthetic data
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, n_basis, wl, rng=42)
    filters = generate_filters(n_filters, wl)
    phot = compute_photometry(spectra, filters, wl)

    # Reconstruction
    P = build_photometry_matrix(build_basis(wl, n_basis), filters, wl)
    coeff_est = reconstruct_coefficients(phot, P, alpha=0.1)
    spectra_rec = reconstruct_spectra(coeff_est, build_basis(wl, n_basis))

    # Simple evaluation
    mse = np.mean((spectra - spectra_rec)**2)
    print(f"Mean squared error of reconstruction: {mse:.4e}")