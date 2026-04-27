import numpy as np
from sklearn.linear_model import Ridge

def generate_wavelength_grid(n_pixels=200, wavelength_min=400, wavelength_max=800):
    """Create a linear wavelength grid."""
    return np.linspace(wavelength_min, wavelength_max, n_pixels)

def create_gaussian_basis(n_basis, wavelengths):
    """Generate a set of Gaussian basis functions."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    widths = 10.0  # fixed width
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        basis.append(g)
    return np.vstack(basis).T  # shape (n_pixels, n_basis)

def generate_random_coefficients(n_basis, rng=np.random.default_rng()):
    """Draw random non‑negative coefficients."""
    return rng.uniform(0.5, 1.5, size=n_basis)

def synthesize_spectrum(basis, coeffs):
    """Linear combination of basis functions."""
    return basis @ coeffs

def generate_filters(n_filters, wavelengths, rng=np.random.default_rng()):
    """Random but smooth filter transmission curves."""
    filters = []
    for _ in range(n_filters):
        amp = rng.uniform(0.5, 1.0, size=n_filters)
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform(20, 80)
        f = amp * np.exp(-0.5 * ((wavelengths - center) / width)**2)
        f /= f.sum()  # normalize
        filters.append(f)
    return np.vstack(filters)  # shape (n_filters, n_pixels)

def photometry_from_spectrum(spectrum, filters):
    """Integrate spectrum through each filter."""
    return filters @ spectrum

def reconstruct_coeffs(photometry, filters, basis, alpha=0.01):
    """Solve for basis coefficients given photometry."""
    # Build matrix that maps coefficients to photometry
    # photometry = filters @ basis @ coeffs
    A = filters @ basis  # shape (n_filters, n_basis)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, photometry)
    return ridge.coef_

def reconstruct_spectrum_from_photometry(photometry, filters, basis, alpha=0.01):
    coeffs = reconstruct_coeffs(photometry, filters, basis, alpha)
    return synthesize_spectrum(basis, coeffs), coeffs

def main():
    rng = np.random.default_rng(42)

    # 1. Define spectral model
    wavelengths = generate_wavelength_grid()
    n_basis = 10
    basis = create_gaussian_basis(n_basis, wavelengths)

    # 2. Generate synthetic spectra
    true_coeffs = generate_random_coefficients(n_basis, rng)
    true_spectrum = synthesize_spectrum(basis, true_coeffs)

    # 3. Generate photometric data
    n_filters = 5
    filters = generate_filters(n_filters, wavelengths, rng)
    photometry = photometry_from_spectrum(true_spectrum, filters)

    # 4. Reconstruct spectrum from photometry
    reconstructed_spectrum, recon_coeffs = reconstruct_spectrum_from_photometry(
        photometry, filters, basis, alpha=0.05)

    # Output results
    print("True coefficients:", true_coeffs)
    print("Recovered coefficients:", recon_coeffs)
    print("Mean squared error:", np.mean((true_spectrum - reconstructed_spectrum)**2))

if __name__ == "__main__":
    main()