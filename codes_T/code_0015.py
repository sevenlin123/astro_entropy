import numpy as np
from scipy.special import erfc
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Spectral model – set of basis functions in wavelength space
# ----------------------------------------------------------------------
def generate_basis_functions(wavelengths, n_basis):
    """Return an array of shape (len(wavelengths), n_basis) containing
    Gaussian basis functions centered uniformly across the wavelength range."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    widths = (wavelengths[-1] - wavelengths[0]) / (2 * n_basis)
    X = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths)**2)
    return X

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra from random coefficients
# ----------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, wavelengths, n_basis):
    """Return a matrix of shape (n_samples, len(wavelengths))."""
    basis = generate_basis_functions(wavelengths, n_basis)
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = coeffs @ basis.T          # linear combination
    spectra += 0.1 * np.random.randn(*spectra.shape)  # add noise
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Create filter transmission curves
# ----------------------------------------------------------------------
def gaussian_filter(wavelengths, center, width):
    """Return transmission curve sampled at wavelengths."""
    return np.exp(-0.5 * ((wavelengths - center) / width)**2)

def generate_filters(wavelengths, n_filters):
    """Generate a set of Gaussian filters evenly spaced over the range."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_filters)
    width = (wavelengths[-1] - wavelengths[0]) / (4 * n_filters)
    filters = [gaussian_filter(wavelengths, c, width) for c in centers]
    return np.array(filters)   # shape (n_filters, len(wavelengths))

# ----------------------------------------------------------------------
# 4. Compute photometry (integrated flux through filters)
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters, wavelengths):
    """Return integrated fluxes for each spectrum and filter."""
    fluxes = spectra @ filters.T      # (n_samples, n_filters)
    return fluxes

# ----------------------------------------------------------------------
# 5. Reconstruction framework – linear mapping from photometry to spectrum
# ----------------------------------------------------------------------
def train_reconstruction_model(photometry, spectra):
    """Fit a linear regression model that maps photometry → spectral coefficients."""
    # Build design matrix: each photometric measurement is a linear combination
    # of basis functions weighted by unknown coefficients.
    # We first compute how each basis function contributes to each filter:
    #   integral(basis_j * filter_i) = A_ij
    # The relation between coefficients c_j and measured flux f_i is:
    #   f_i = sum_j c_j * A_ij + noise
    # Thus A is the matrix mapping coeffs -> photometry.
    # Inverse problem: find coeffs from f_i.
    return

# For simplicity, we directly solve for coefficients using least squares:
def reconstruct_coefficients(photometry, filter_matrix, basis):
    """
    photometry: (n_samples, n_filters)
    filter_matrix: (n_filters, len(wavelengths))
    basis: (len(wavelengths), n_basis)
    Returns estimated coefficients of shape (n_samples, n_basis)
    """
    # Compute convolution matrix: for each basis function j, its
    # contribution to each filter i:
    conv = filter_matrix @ basis   # (n_filters, n_basis)
    # Solve linear system conv^T c = f for each sample
    coeffs = np.linalg.lstsq(conv.T, photometry.T, rcond=None)[0].T
    return coeffs

# ----------------------------------------------------------------------
# 6. Reconstruct spectra from photometry
# ----------------------------------------------------------------------
def reconstruct_spectrum_from_photometry(photometry, filter_matrix, basis):
    coeffs = reconstruct_coefficients(photometry, filter_matrix, basis)
    reconstructed = coeffs @ basis.T
    return reconstructed

# ----------------------------------------------------------------------
# 7. Main routine – generate data, train, evaluate
# ----------------------------------------------------------------------
def main():
    # Wavelength grid
    wav_min, wav_max, n_wave = 300.0, 900.0, 601
    wavelengths = np.linspace(wav_min, wav_max, n_wave)

    # Hyperparameters
    n_basis = 10
    n_filters = 5
    n_train = 200
    n_test = 20

    # Generate synthetic training spectra and coefficients
    train_spectra, train_coeffs = generate_synthetic_spectra(n_train, wavelengths, n_basis)

    # Generate filters
    filters = generate_filters(wavelengths, n_filters)   # shape (n_filters, n_wave)

    # Compute training photometry
    train_flux = compute_photometry(train_spectra, filters, wavelengths)

    # Reconstruct spectra from training photometry
    recon_train = reconstruct_spectrum_from_photometry(train_flux, filters, generate_basis_functions(wavelengths, n_basis))

    # Evaluate training error
    train_error = np.mean((train_spectra - recon_train)**2)
    print(f"Training MSE: {train_error:.6f}")

    # Test data
    test_spectra, test_coeffs = generate_synthetic_spectra(n_test, wavelengths, n_basis)
    test_flux = compute_photometry(test_spectra, filters, wavelengths)
    recon_test = reconstruct_spectrum_from_photometry(test_flux, filters, generate_basis_functions(wavelengths, n_basis))

    # Evaluate test error
    test_error = np.mean((test_spectra - recon_test)**2)
    print(f"Test MSE: {test_error:.6f}")

if __name__ == "__main__":
    main()