import numpy as np
from sklearn.linear_model import LinearRegression

def create_wavelength_grid(start=400, stop=1000, step=5):
    """Generate a wavelength grid (in nm)."""
    return np.arange(start, stop + step, step)

def gaussian_basis(wl, center, width):
    """Single Gaussian basis function."""
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def build_basis_functions(wl, n_bases=10, width=20):
    """Construct a set of Gaussian basis functions."""
    centers = np.linspace(wl[0], wl[-1], n_bases)
    basis = np.vstack([gaussian_basis(wl, c, width) for c in centers]).T
    return basis, centers

def generate_synthetic_spectra(basis, n_spectra=50, sigma=0.1):
    """Generate synthetic spectra as random linear combinations of basis."""
    rng = np.random.default_rng(seed=42)
    coeffs = rng.normal(0, sigma, size=(n_spectra, basis.shape[1]))
    spectra = coeffs @ basis.T
    return spectra, coeffs

def rectangle_filter(wl, low, high):
    """Simple rectangular filter transmission."""
    return ((wl >= low) & (wl <= high)).astype(float)

def build_filters(wl):
    """Define a set of broadband filters."""
    filter_defs = [(450, 550), (550, 650), (650, 750), (750, 850)]
    filters = np.array([rectangle_filter(wl, lo, hi) for lo, hi in filter_defs])
    return filters, [f"{lo}-{hi}" for lo, hi in filter_defs]

def compute_photometry(spectra, filters):
    """Compute integrated fluxes through each filter."""
    # Assuming equal spacing in wl, so integration is sum over wl
    return spectra @ filters.T

def reconstruct_coefficients(photometry, basis, filters):
    """Reconstruct coefficients using linear regression on filter responses."""
    # Build design matrix: for each filter, integrate each basis over filter
    M = np.sum(basis[:, :, None] * filters[None, :, :], axis=1)  # shape (filters, basis)
    # Solve least-squares problem
    reg = LinearRegression(fit_intercept=False)
    reg.fit(M, photometry.T)
    return reg.coef_.T

def reconstruct_spectrum(coeffs, basis):
    """Reconstruct spectra from coefficients."""
    return coeffs @ basis.T

if __name__ == "__main__":
    # 1. Define spectral model
    wl = create_wavelength_grid()
    basis, centers = build_basis_functions(wl)

    # 2. Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(basis)

    # 3. Generate photometric data
    filters, filter_names = build_filters(wl)
    photometry = compute_photometry(spectra_true, filters)

    # 4. Reconstruct spectra
    coeffs_rec = reconstruct_coefficients(photometry, basis, filters)
    spectra_rec = reconstruct_spectrum(coeffs_rec, basis)

    # Evaluate reconstruction
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec) ** 2))
    print(f"RMSE between true and reconstructed spectra: {rmse:.4f}")

    # Optional: compare coefficients
    coeff_rmse = np.sqrt(np.mean((coeffs_true - coeffs_rec) ** 2))
    print(f"Coefficient RMSE: {coeff_rmse:.4f}")