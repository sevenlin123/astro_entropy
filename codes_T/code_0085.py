import numpy as np
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #

def generate_wavelength_grid(n_lambda=1000, lam_min=3500, lam_max=8000):
    """Create a regular wavelength grid (in Å)."""
    return np.linspace(lam_min, lam_max, n_lambda)

def generate_basis_spectra(n_basis, n_lambda, rng):
    """
    Create a set of smooth basis spectra.
    Each basis spectrum is a random noise convolved with a Gaussian kernel.
    """
    raw = rng.randn(n_basis, n_lambda)
    kernel = np.exp(-np.linspace(-3, 3, 15)**2 / 0.5**2)
    kernel /= kernel.sum()
    basis = np.array([convolve(row, kernel, mode='same') for row in raw])
    # Normalize each basis spectrum to unit mean
    basis /= basis.mean(axis=1, keepdims=True)
    return basis  # shape (n_basis, n_lambda)

def spectral_model(params, basis_spectra):
    """Linear combination of basis spectra with given coefficients."""
    return np.dot(params, basis_spectra)  # shape (n_lambda,)

def generate_filter_transmissions(n_filters, n_lambda, rng):
    """
    Generate simple top‑hat filter transmissions.
    Each filter is centered at a random wavelength and has a fixed width.
    """
    lam_center = rng.uniform(low=4000, high=7500, size=n_filters)
    width = 200.0  # Å
    filters = np.zeros((n_filters, n_lambda))
    for i, lc in enumerate(lam_center):
        mask = np.abs(np.linspace(4000, 7500, n_lambda) - lc) < width / 2
        filters[i, mask] = 1.0
    return filters  # shape (n_filters, n_lambda)

def compute_photometry(spectrum, filters):
    """Integrate spectrum over each filter to get photometric fluxes."""
    return np.dot(filters, spectrum)  # shape (n_filters,)

def reconstruct_coefficients(photon_flux, filter_matrix, n_basis):
    """
    Estimate the coefficients of the basis spectra from photometric data.
    Uses ordinary least squares (scikit‑learn LinearRegression).
    """
    lr = LinearRegression(fit_intercept=False)
    lr.fit(filter_matrix, photon_flux)
    return lr.coef_  # shape (n_basis,)

def reconstruct_spectrum_from_coefficients(coeffs, basis_spectra):
    """Reconstruct the full spectrum from estimated coefficients."""
    return spectral_model(coeffs, basis_spectra)  # shape (n_lambda,)

# --------------------------------------------------------------------------- #
# Demo workflow
# --------------------------------------------------------------------------- #

def main():
    rng = np.random.default_rng(seed=42)

    # Wavelength grid
    n_lambda = 1000
    lam_grid = generate_wavelength_grid(n_lambda=n_lambda)

    # Basis spectra
    n_basis = 5
    basis = generate_basis_spectra(n_basis=n_basis, n_lambda=n_lambda, rng=rng)

    # Filter transmissions
    n_filters = 4
    filters = generate_filter_transmissions(n_filters=n_filters, n_lambda=n_lambda, rng=rng)

    # Precompute filter responses for each basis spectrum
    # Response matrix R: shape (n_filters, n_basis)
    R = np.zeros((n_filters, n_basis))
    for j in range(n_basis):
        R[:, j] = compute_photometry(basis[j], filters)

    # Generate synthetic spectra and photometry
    n_objects = 10
    true_params = rng.uniform(0.5, 2.0, size=(n_objects, n_basis))
    spectra_true = np.array([spectral_model(p, basis) for p in true_params])

    photometry = np.array([compute_photometry(s, filters) for s in spectra_true])

    # Reconstruct spectra from photometry
    spectra_recon = []
    errors = []
    for idx in range(n_objects):
        p = photometry[idx]
        coeff_est = reconstruct_coefficients(p, R, n_basis)
        s_hat = reconstruct_spectrum_from_coefficients(coeff_est, basis)
        spectra_recon.append(s_hat)
        err = np.linalg.norm(s_hat - spectra_true[idx]) / np.linalg.norm(spectra_true[idx])
        errors.append(err)

    # Output results
    print("Mean relative reconstruction error:", np.mean(errors))
    for i in range(3):
        print(f"Object {i+1} reconstruction error: {errors[i]:.4f}")

if __name__ == "__main__":
    main()