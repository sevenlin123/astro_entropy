import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.integrate import trapz

def create_wavelength_grid(start=400, stop=700, n_points=100):
    """Create wavelength grid in nanometers."""
    return np.linspace(start, stop, n_points)

def gaussian_filter(wl, center, width):
    """Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wl - center) / width)**2)

def build_filters(wl, centers, width=20):
    """Build filter transmission curves for given centers."""
    return [gaussian_filter(wl, c, width) for c in centers]

def basis_functions(wl, n_basis=10):
    """Generate synthetic spectral basis functions."""
    rng = np.random.default_rng(seed=42)
    basis = []
    for _ in range(n_basis):
        # random smooth function via spline interpolation of random points
        knots = np.linspace(wl.min(), wl.max(), 8)
        values = rng.uniform(0.1, 1.0, size=knots.shape)
        f = interp1d(knots, values, kind='cubic', bounds_error=False,
                     fill_value="extrapolate")
        basis.append(f(wl))
    return np.vstack(basis)  # shape (n_basis, n_wl)

def synthesize_spectrum(basis, n_realizations=50):
    """Generate synthetic spectra as random linear combinations of basis."""
    rng = np.random.default_rng(seed=24)
    coeffs = rng.normal(size=(n_realizations, basis.shape[0]))
    spectra = coeffs @ basis
    # Add small Gaussian noise
    spectra += rng.normal(scale=0.01, size=spectra.shape)
    return spectra, coeffs

def compute_photometry(spectra, wl, filters):
    """Integrate spectra over filter transmission curves."""
    phot = []
    for filt in filters:
        trans = filt
        phot.append(trapz(spectra.T * trans, wl, axis=0))
    return np.vstack(phot).T  # shape (n_samples, n_filters)

def build_forward_matrix(basis, wl, filters):
    """Compute filter responses for each basis function."""
    responses = []
    for filt in filters:
        resp = trapz(basis.T * filt, wl, axis=1)  # shape (n_basis,)
        responses.append(resp)
    return np.column_stack(responses)  # shape (n_basis, n_filters)

def reconstruct_spectra(phot, forward_matrix, basis):
    """Reconstruct spectra from photometry using linear regression."""
    # Solve for coefficients via least squares
    reg = LinearRegression(fit_intercept=False)
    reg.fit(forward_matrix.T, phot.T)  # forward_matrix^T shape (n_filters, n_basis)
    coeffs_pred = reg.coef_.T
    reconstructed = coeffs_pred @ basis
    return reconstructed, coeffs_pred

def main():
    # 1. Define wavelength grid
    wl = create_wavelength_grid()

    # 2. Build spectral basis
    basis = basis_functions(wl, n_basis=12)  # (12, n_wl)

    # 3. Generate synthetic spectra
    spectra, true_coeffs = synthesize_spectrum(basis, n_realizations=200)  # (200, n_wl)

    # 4. Define photometric filters
    filter_centers = [450, 500, 550, 600, 650]
    filters = build_filters(wl, filter_centers, width=30)

    # 5. Compute photometry from synthetic spectra
    phot = compute_photometry(spectra, wl, filters)  # (200, 5)

    # 6. Build forward matrix for reconstruction
    forward_matrix = build_forward_matrix(basis, wl, filters)  # (12, 5)

    # 7. Reconstruct spectra from photometry
    recon_spectra, recon_coeffs = reconstruct_spectra(phot, forward_matrix, basis)

    # 8. Evaluate reconstruction error
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Reconstruction MSE: {mse:.6f}")

if __name__ == "__main__":
    main()