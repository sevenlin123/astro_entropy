import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------
# Spectral model and utilities
# ----------------------------
def generate_wavelength_grid(start=300, end=800, step=5):
    """Wavelength grid in nanometers."""
    return np.arange(start, end + step, step)

def gaussian_basis(wavelengths, center, width):
    """Single Gaussian basis spectrum."""
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def generate_basis_spectra(n_basis, wavelengths):
    """Generate a set of Gaussian basis spectra."""
    np.random.seed(0)
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = np.full(n_basis, 30.0)
    basis = [gaussian_basis(wavelengths, c, w) for c, w in zip(centers, widths)]
    return np.array(basis)  # shape (n_basis, n_wave)

def generate_synthetic_spectrum(coeffs, basis_spectra):
    """Linear combination of basis spectra."""
    return coeffs @ basis_spectra  # shape (n_wave,)

def rectangular_filter(wavelengths, band_center, bandwidth):
    """Simple top-hat filter transmission."""
    lower = band_center - bandwidth / 2
    upper = band_center + bandwidth / 2
    return ((wavelengths >= lower) & (wavelengths <= upper)).astype(float)

def generate_filters(n_filters, wavelengths):
    """Generate a set of rectangular photometric filters."""
    np.random.seed(1)
    centers = np.linspace(wavelengths.min() + 50, wavelengths.max() - 50, n_filters)
    bandwidth = 80.0
    filters = [rectangular_filter(wavelengths, c, bandwidth) for c in centers]
    return np.array(filters)  # shape (n_filters, n_wave)

# ----------------------------
# Photometry and reconstruction
# ----------------------------
def compute_photometry(spectrum, filters):
    """Integrate spectrum over each filter."""
    return np.array([np.trapz(spectrum * f, x=f.shape[0]) for f in filters])

def build_design_matrix(basis_spectra, filters):
    """
    Construct matrix A where A[j,i] = integral(basis_i * filter_j).
    Filters and basis_spectra are arrays of shape (n_filters, n_wave) and
    (n_basis, n_wave) respectively.
    """
    n_filters, n_wave = filters.shape
    n_basis = basis_spectra.shape[0]
    A = np.empty((n_filters, n_basis))
    for j in range(n_filters):
        for i in range(n_basis):
            A[j, i] = np.trapz(basis_spectra[i] * filters[j], x=np.arange(n_wave))
    return A

def reconstruct_spectrum_from_photometry(photometry, basis_spectra, filters):
    """Recover spectrum coefficients via least-squares."""
    A = build_design_matrix(basis_spectra, filters)
    reg = LinearRegression(fit_intercept=False).fit(A, photometry)
    coeffs = reg.coef_
    reconstructed = coeffs @ basis_spectra
    return reconstructed, coeffs

# ----------------------------
# Demo
# ----------------------------
def main():
    wavelengths = generate_wavelength_grid()
    n_basis = 5
    basis = generate_basis_spectra(n_basis, wavelengths)
    
    # True coefficients
    np.random.seed(42)
    true_coeffs = np.random.uniform(0.5, 1.5, size=n_basis)
    
    true_spectrum = generate_synthetic_spectrum(true_coeffs, basis)
    
    n_filters = 4
    filters = generate_filters(n_filters, wavelengths)
    photometry = compute_photometry(true_spectrum, filters)
    
    recon_spectrum, recon_coeffs = reconstruct_spectrum_from_photometry(
        photometry, basis, filters)
    
    print("True coefficients:", true_coeffs)
    print("Reconstructed coefficients:", recon_coeffs)
    error = np.linalg.norm(true_spectrum - recon_spectrum) / np.linalg.norm(true_spectrum)
    print("Relative reconstruction error:", error)

if __name__ == "__main__":
    main()