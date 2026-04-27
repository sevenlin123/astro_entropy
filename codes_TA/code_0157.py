import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import Ridge

def create_wavelength_grid(start=400.0, stop=700.0, num=500):
    """Create an equidistant wavelength grid in nanometers."""
    return np.linspace(start, stop, num)

def gaussian_basis(wavelengths, centers, width=20.0):
    """Generate Gaussian basis functions evaluated on the wavelength grid."""
    basis = []
    for c in centers:
        basis.append(np.exp(-0.5 * ((wavelengths - c) / width) ** 2))
    return np.vstack(basis).T  # shape (len(wavelengths), len(centers))

def gaussian_filter(wavelengths, center, width=30.0):
    """Generate a single Gaussian filter curve."""
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)

def create_filters(wavelengths, centers, width=30.0):
    """Generate a set of filter transmission curves."""
    filters = [gaussian_filter(wavelengths, c, width) for c in centers]
    return np.vstack(filters).T  # shape (len(filters), len(wavelengths))

def generate_synthetic_spectrum(basis, coeffs):
    """Linear combination of basis functions with given coefficients."""
    return basis @ coeffs  # shape (len(wavelengths),)

def compute_photometry(spectrum, filters, wavelengths):
    """Integrate spectrum through each filter to produce photometric fluxes."""
    photometry = []
    for filt in filters:
        # Integrate flux * filter transmission over wavelength
        photometry.append(trapz(spectrum * filt, wavelengths))
    return np.array(photometry)

def reconstruct_coefficients(filters, basis, photometry, alpha=1e-2):
    """Recover coefficients via ridge regression from filter integrals."""
    # Build design matrix: each row = integral of basis_j * filter_i
    num_filters, num_wavelengths = filters.shape
    num_basis = basis.shape[1]
    M = np.zeros((num_filters, num_basis))
    for i in range(num_filters):
        for j in range(num_basis):
            M[i, j] = trapz(basis[:, j] * filters[i], axis=1).sum()
    # Fit ridge regression (no intercept)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(M, photometry)
    return ridge.coef_

def main():
    np.random.seed(42)
    # Wavelength grid
    wav = create_wavelength_grid()
    # Basis functions (Gaussians)
    basis_centers = np.linspace(420, 680, 4)
    basis = gaussian_basis(wav, basis_centers, width=25.0)
    # Filters (Gaussians)
    filter_centers = np.linspace(450, 650, 6)
    filters = create_filters(wav, filter_centers, width=35.0)
    # True coefficients for synthetic spectrum
    true_coeffs = np.random.uniform(0.5, 1.5, size=basis.shape[1])
    # Generate synthetic spectrum
    spectrum_true = generate_synthetic_spectrum(basis, true_coeffs)
    # Compute photometry
    photometry = compute_photometry(spectrum_true, filters, wav)
    # Reconstruct coefficients
    coeffs_rec = reconstruct_coefficients(filters, basis, photometry, alpha=0.01)
    # Reconstruct spectrum
    spectrum_rec = generate_synthetic_spectrum(basis, coeffs_rec)
    # Evaluation
    rms_error = np.sqrt(np.mean((spectrum_true - spectrum_rec) ** 2))
    print("True coefficients:", true_coeffs)
    print("Recovered coefficients:", coeffs_rec)
    print(f"RMS reconstruction error: {rms_error:.4f}")

if __name__ == "__main__":
    main()