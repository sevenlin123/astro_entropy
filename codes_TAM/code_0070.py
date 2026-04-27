import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Spectral model utilities
# ------------------------------------------------------------------
def create_wavelength_grid(n_points=200, lam_min=4000, lam_max=8000):
    """Create a uniform wavelength grid."""
    return np.linspace(lam_min, lam_max, n_points)

def generate_basis_spectra(n_basis, wavelengths):
    """Generate synthetic basis spectra (e.g., atomic lines + continuum)."""
    n_points = len(wavelengths)
    basis = []
    rng = np.random.default_rng(seed=42)
    # Continuum component
    continuum = rng.normal(loc=0.5, scale=0.05, size=n_points)
    basis.append(continuum)
    # Gaussian line components
    for i in range(n_basis - 1):
        amp = rng.uniform(0.05, 0.2)
        sigma = rng.uniform(10, 30)
        mu = rng.uniform(wavelengths[0], wavelengths[-1])
        line = amp * np.exp(-0.5 * ((wavelengths - mu) / sigma)**2)
        basis.append(line)
    return np.vstack(basis)  # shape (n_basis, n_points)

def generate_random_coefficients(n_basis):
    """Random non‑negative coefficients for a linear combination."""
    rng = np.random.default_rng()
    coeffs = rng.uniform(0.8, 1.2, size=n_basis)
    return coeffs

def build_spectrum(basis, coeffs):
    """Linear combination of basis spectra."""
    return np.dot(coeffs, basis)  # shape (n_points,)

# ------------------------------------------------------------------
# Photometry utilities
# ------------------------------------------------------------------
def gaussian_filter_profile(wavelengths, mu, sigma):
    """Gaussian filter transmission profile."""
    return np.exp(-0.5 * ((wavelengths - mu) / sigma)**2)

def generate_filters(n_filters, wavelengths):
    """Generate simple Gaussian filter profiles."""
    rng = np.random.default_rng(seed=99)
    filters = []
    for _ in range(n_filters):
        mu = rng.uniform(wavelengths[0] + 500, wavelengths[-1] - 500)
        sigma = rng.uniform(100, 300)
        trans = gaussian_filter_profile(wavelengths, mu, sigma)
        filters.append(trans / trans.sum())  # normalise
    return filters  # list of arrays length n_points

def compute_photometry(spectrum, filters):
    """Integrate spectrum through each filter."""
    phot = [np.sum(spectrum * f) for f in filters]
    return np.array(phot)  # shape (n_filters,)

# ------------------------------------------------------------------
# Reconstruction utilities
# ------------------------------------------------------------------
def build_photometry_matrix(basis, filters):
    """Build matrix mapping basis coefficients to photometry."""
    n_filters = len(filters)
    n_basis = basis.shape[0]
    A = np.zeros((n_filters, n_basis))
    for j, filt in enumerate(filters):
        for i, b in enumerate(basis):
            A[j, i] = np.sum(b * filt)
    return A  # shape (n_filters, n_basis)

def reconstruct_coefficients(photometry, basis, filters):
    """Least‑squares estimation of basis coefficients."""
    A = build_photometry_matrix(basis, filters)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(A, photometry)
    return lr.coef_  # shape (n_basis,)

def reconstruct_spectrum_from_photometry(photometry, basis, filters):
    coeffs = reconstruct_coefficients(photometry, basis, filters)
    return build_spectrum(basis, coeffs), coeffs

# ------------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------------
def main():
    # Create wavelength grid
    wav = create_wavelength_grid()

    # Generate basis spectra
    n_basis = 5
    basis = generate_basis_spectra(n_basis, wav)

    # Generate synthetic spectrum
    true_coeffs = generate_random_coefficients(n_basis)
    true_spectrum = build_spectrum(basis, true_coeffs)

    # Generate filters
    n_filters = 3
    filters = generate_filters(n_filters, wav)

    # Compute photometric measurements
    photometry = compute_photometry(true_spectrum, filters)

    # Reconstruct spectrum from photometry
    reconstructed_spectrum, recon_coeffs = reconstruct_spectrum_from_photometry(
        photometry, basis, filters
    )

    # Simple diagnostics
    print("True coefficients :", true_coeffs)
    print("Recovered coeffs :", recon_coeffs)
    print("Spectral difference RMS:", np.sqrt(np.mean((true_spectrum - reconstructed_spectrum)**2)))

if __name__ == "__main__":
    main()