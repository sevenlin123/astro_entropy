import numpy as np
from sklearn.linear_model import LinearRegression

def create_wavelength_grid():
    """Create a linear wavelength grid from 400 to 700 nm."""
    return np.linspace(400, 700, 50)

def create_basis_functions(grid, n_basis=5):
    """Generate Gaussian basis functions."""
    centers = np.linspace(grid[0], grid[-1], n_basis + 2)[1:-1]
    widths = (grid[-1] - grid[0]) / (n_basis * 4)
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((grid - c) / widths)**2)
        basis.append(g)
    return np.array(basis)  # shape (n_basis, len(grid))

def evaluate_spectrum(coeffs, basis_funcs, grid):
    """Evaluate a spectrum given coefficients and basis functions."""
    return coeffs @ basis_funcs  # shape (len(grid),)

def generate_synthetic_spectra(n_samples, basis_funcs, grid, noise_level=0.02):
    """Generate synthetic spectra with random coefficients."""
    n_basis = basis_funcs.shape[0]
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = []
    for i in range(n_samples):
        spec = evaluate_spectrum(coeffs[i], basis_funcs, grid)
        spec += noise_level * np.random.randn(len(grid))
        spectra.append(spec)
    return np.array(spectra), coeffs

def define_filters(grid):
    """Define simple top‑hat photometric filters."""
    filt_edges = [(400, 450), (450, 500), (500, 550),
                  (550, 600), (600, 650)]
    filters = []
    for lo, hi in filt_edges:
        filt = np.where((grid >= lo) & (grid <= hi), 1.0, 0.0)
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, len(grid))

def compute_photometry(spectra, filters, noise_rel=0.05):
    """Compute photometric fluxes from spectra."""
    n_samples, _ = spectra.shape
    n_filters = filters.shape[0]
    phot = np.zeros((n_samples, n_filters))
    for i in range(n_samples):
        for j in range(n_filters):
            filt = filters[j]
            flux = np.trapz(spectra[i] * filt, dx=filters.shape[1]/(filters.shape[1]-1))
            norm = np.trapz(filt, dx=filters.shape[1]/(filters.shape[1]-1))
            phot[i, j] = flux / norm
    # Add relative Gaussian noise
    phot += noise_rel * phot * np.random.randn(*phot.shape)
    return phot

def reconstruct_coeffs(photometry, basis_funcs, filters):
    """Reconstruct spectral coefficients from photometry."""
    n_filters, _ = filters.shape
    n_basis = basis_funcs.shape[0]
    # Build the filter‑basis integral matrix A (n_filters, n_basis)
    A = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        filt = filters[i]
        norm = np.trapz(filt, dx=filters.shape[1]/(filters.shape[1]-1))
        for j in range(n_basis):
            A[i, j] = np.trapz(basis_funcs[j] * filt, dx=filters.shape[1]/(filters.shape[1]-1)) / norm
    # Solve least‑squares for each spectrum
    coeffs_rec = np.linalg.lstsq(A, photometry.T, rcond=None)[0].T
    return coeffs_rec

def reconstruct_spectra_from_coeffs(coeffs, basis_funcs, grid):
    """Reconstruct spectra from recovered coefficients."""
    return coeffs @ basis_funcs  # shape (n_samples, len(grid))

# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    grid = create_wavelength_grid()
    basis_funcs = create_basis_functions(grid, n_basis=5)
    spectra_true, coeffs_true = generate_synthetic_spectra(
        n_samples=10, basis_funcs=basis_funcs, grid=grid
    )
    filters = define_filters(grid)
    photometry = compute_photometry(spectra_true, filters)
    coeffs_rec = reconstruct_coeffs(photometry, basis_funcs, filters)
    spectra_rec = reconstruct_spectra_from_coeffs(coeffs_rec, basis_funcs, grid)

    # Evaluate reconstruction error
    mse = np.mean((spectra_true - spectra_rec)**2, axis=1)
    print("Mean squared reconstruction error per spectrum:")
    for i, e in enumerate(mse):
        print(f"  Spectrum {i}: {e:.4f}")