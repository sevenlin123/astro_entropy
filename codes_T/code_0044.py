import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# 1. Define a spectral model
def spectral_model(wavelengths, coeffs, basis):
    """Generate a spectrum as a linear combination of basis functions."""
    return np.dot(coeffs, basis(wavelengths))

def gaussian_basis(wavelengths, centers, widths):
    """Create Gaussian basis functions."""
    g = []
    for c, w in zip(centers, widths):
        g.append(np.exp(-0.5 * ((wavelengths - c) / w)**2))
    return np.array(g)

# 2. Generate synthetic spectra
def generate_synthetic_spectrum(n_samples=50, n_bases=10, wavelength_range=(400, 700)):
    wavelengths = np.linspace(*wavelength_range, 300)
    centers = np.linspace(wavelength_range[0], wavelength_range[1], n_bases)
    widths = np.full(n_bases, 15.0)
    basis = lambda wl: gaussian_basis(wl, centers, widths)
    coeffs = np.random.randn(n_samples, n_bases)
    spectra = np.array([spectral_model(wavelengths, c, basis) for c in coeffs])
    return wavelengths, spectra, coeffs, basis

# 3. generate photometric data from synthetic spectra
def photometric_filter(wavelengths, filter_center, filter_width):
    """Gaussian filter response."""
    response = np.exp(-0.5 * ((wavelengths - filter_center) / filter_width)**2)
    return response / np.sum(response)

def photometric_measurement(spectrum, wavelengths, filters):
    """Integrate spectrum over each filter."""
    measurements = []
    for filt in filters:
        response = photometric_filter(wavelengths, filt['center'], filt['width'])
        flux = np.trapz(spectrum * response, wavelengths)
        measurements.append(flux)
    return np.array(measurements)

def generate_photometry(spectra, wavelengths, n_filters=3):
    np.random.seed(42)
    filters = [{'center': np.random.uniform(450, 650), 'width': np.random.uniform(20, 40)}
               for _ in range(n_filters)]
    photom = np.array([photometric_measurement(s, wavelengths, filters) for s in spectra])
    return photom, filters

# 4. reconstruct a synthetic spectrum from photometruic
def reconstruct_spectrum_from_photometry(photom, wavelengths, filters, n_bases=10):
    # Build design matrix A from filter responses convolved with basis
    A = []
    for filt in filters:
        resp = photometric_filter(wavelengths, filt['center'], filt['width'])
        # Convolve each basis function with filter response
        row = [np.trapz(gaussian_basis(wavelengths, [c], [w]) * resp, wavelengths)
               for c, w in zip(np.linspace(400, 700, n_bases), np.full(n_bases, 15.0))]
        A.append(row)
    A = np.array(A).T  # shape (n_bases, n_filters)
    # Solve for coefficients using ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(A.T, photom)
    return ridge.coef_.T  # coefficients for each sample

# 5. Demo
if __name__ == "__main__":
    wavelengths, spectra, true_coeffs, basis_func = generate_synthetic_spectrum()
    photom, filters = generate_photometry(spectra, wavelengths)
    estimated_coeffs = reconstruct_spectrum_from_photometry(photom, wavelengths, filters)
    # Reconstruct spectra
    reconstructed_spectra = np.array([spectral_model(wavelengths, c, basis_func) for c in estimated_coeffs])
    print("True coefficient matrix shape:", true_coeffs.shape)
    print("Estimated coefficient matrix shape:", estimated_coeffs.shape)
    # Show difference
    err = np.mean((true_coeffs - estimated_coeffs)**2)
    print(f"Mean squared error of coefficients: {err:.4f}")