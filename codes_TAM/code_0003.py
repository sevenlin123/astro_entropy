import numpy as np
from sklearn.linear_model import LinearRegression

def wavelength_grid(start=3000, stop=10000, num=500):
    """Return a linear wavelength grid in Å."""
    return np.linspace(start, stop, num)

def basis_functions(wl, n_basis=10):
    """Generate a set of orthogonal basis functions."""
    basis = []
    for k in range(1, n_basis + 1):
        # sine waves with increasing frequency
        basis.append(np.sin(k * np.pi * (wl - wl[0]) / (wl[-1] - wl[0])))
    return np.array(basis)  # shape (n_basis, n_wl)

def synthetic_spectrum(basis, rng=np.random.default_rng()):
    """Create a synthetic spectrum as a random linear combination of basis."""
    coeffs = rng.uniform(-1, 1, size=basis.shape[0])
    spec = coeffs @ basis  # shape (n_wl,)
    return spec, coeffs

def gaussian_filter(wl, center, width, amplitude=1.0):
    """Return a Gaussian filter response."""
    return amplitude * np.exp(-0.5 * ((wl - center) / width) ** 2)

def filter_set(wl, centers=[4000, 5000, 6000, 7000, 8000], width=200):
    """Generate a set of filter responses."""
    return np.vstack([gaussian_filter(wl, c, width) for c in centers])

def compute_photometry(spectrum, filters, wl_step):
    """
    Integrate spectrum through each filter.
    Spectrum and filters are assumed to be sampled at the same wavelength grid.
    """
    return np.sum(spectrum[:, None] * filters, axis=0) * wl_step

def build_response_matrix(filters, basis, wl_step):
    """
    Build the response matrix M (filters x basis) where each element is the
    integral of a basis function through a filter.
    """
    return np.sum(basis[:, None, :] * filters[None, :, :], axis=2) * wl_step

def reconstruct_spectrum(photometry, filters, basis, wl_step):
    """
    Reconstruct spectrum from photometry using linear regression on the basis.
    """
    M = build_response_matrix(filters, basis, wl_step)
    lr = LinearRegression(fit_intercept=False).fit(M.T, photometry)
    coeffs_hat = lr.coef_
    recon_spec = coeffs_hat @ basis
    return recon_spec, coeffs_hat

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Wavelength grid
    wl = wavelength_grid()
    wl_step = wl[1] - wl[0]

    # Basis functions
    basis = basis_functions(wl, n_basis=15)

    # Synthetic spectrum
    spec_true, coeff_true = synthetic_spectrum(basis, rng=rng)

    # Filter set
    filters = filter_set(wl)

    # Photometry from synthetic spectrum
    phot_true = compute_photometry(spec_true, filters, wl_step)

    # Reconstruction
    spec_rec, coeff_rec = reconstruct_spectrum(phot_true, filters, basis, wl_step)

    # Error metrics
    mae = np.mean(np.abs(spec_true - spec_rec))
    rmse = np.sqrt(np.mean((spec_true - spec_rec) ** 2))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Optional: compare coefficients
    coeff_diff = np.linalg.norm(coeff_true - coeff_rec)
    print(f"Coeff L2 diff: {coeff_diff:.4f}")