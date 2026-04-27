import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------
# 1. Spectral model and synthetic data generation
# -------------------------------------------------------------

def wavelength_grid(start=400.0, stop=800.0, num=200):
    """Generate wavelength array in nm."""
    return np.linspace(start, stop, num)

def gaussian(x, mu, sigma):
    """Simple Gaussian function."""
    return np.exp(-0.5 * ((x - mu) / sigma)**2)

def generate_basis(n_bases, wavelengths):
    """
    Create n_bases Gaussian basis spectra with random centres and widths.
    Returns: (n_bases, len(wavelengths)) array.
    """
    np.random.seed(0)
    mus = np.random.uniform(wavelengths[0], wavelengths[-1], n_bases)
    sigmas = np.random.uniform(20, 60, n_bases)
    basis = np.array([gaussian(wavelengths, mu, sigma) for mu, sigma in zip(mus, sigmas)])
    # normalise each basis spectrum
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    return basis

def generate_synthetic_spectra(n_samples, basis, wavelengths):
    """
    Generate synthetic spectra as random linear combinations of basis functions.
    Returns: (n_samples, len(wavelengths)) array.
    """
    np.random.seed(1)
    coeffs = np.random.normal(size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis
    return spectra, coeffs

# -------------------------------------------------------------
# 2. Photometric filter definition and flux calculation
# -------------------------------------------------------------

def gaussian_filter(x, center, width):
    """Return filter transmission curve as Gaussian."""
    return gaussian(x, center, width)

def define_filters():
    """
    Define a set of three filters (U, V, R) with different centres and widths.
    Each filter returns its transmission curve.
    """
    filters = {
        'U': lambda x: gaussian_filter(x, 350.0, 30.0),
        'V': lambda x: gaussian_filter(x, 550.0, 40.0),
        'R': lambda x: gaussian_filter(x, 700.0, 35.0),
    }
    return filters

def photometry_from_spectra(spectra, wavelengths, filters):
    """
    Compute photometric fluxes for each spectrum and filter.
    Returns: (n_samples, n_filters) array.
    """
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    phot = np.zeros((n_samples, n_filters))
    filter_keys = list(filters.keys())
    for i, filt_name in enumerate(filter_keys):
        trans = filters[filt_name](wavelengths)
        for j in range(n_samples):
            phot[j, i] = simps(spectra[j] * trans, wavelengths)
    return phot, filter_keys

# -------------------------------------------------------------
# 3. Reconstruction from photometry
# -------------------------------------------------------------

def construct_design_matrix(basis, wavelengths, filters):
    """
    Build the matrix that maps basis coefficients to photometric fluxes.
    Shape: (n_filters, n_bases)
    """
    n_filters = len(filters)
    n_bases = basis.shape[0]
    A = np.zeros((n_filters, n_bases))
    filter_keys = list(filters.keys())
    for i, filt_name in enumerate(filter_keys):
        trans = filters[filt_name](wavelengths)
        # integral of basis * filter over wavelength
        A[i, :] = simps(basis.T * trans[:, None], wavelengths, axis=0)
    return A

def reconstruct_coefficients(phot, design_matrix):
    """
    Solve for basis coefficients using least squares.
    Returns: (n_samples, n_bases) array.
    """
    # Using sklearn's LinearRegression for clarity
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix.T, phot.T)
    coeffs_est = reg.coef_.T
    return coeffs_est

def reconstruct_spectra(coeffs_est, basis):
    """
    Reconstruct spectra from estimated coefficients.
    """
    return coeffs_est @ basis

# -------------------------------------------------------------
# 4. Main execution block
# -------------------------------------------------------------

def main():
    # Define wavelength grid
    wavelengths = wavelength_grid()

    # Generate basis spectra
    n_bases = 10
    basis = generate_basis(n_bases, wavelengths)

    # Generate synthetic spectra
    n_samples = 5
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis, wavelengths)

    # Define filters
    filters = define_filters()

    # Compute photometric fluxes
    phot, filter_names = photometry_from_spectra(spectra, wavelengths, filters)

    # Construct design matrix
    design_mat = construct_design_matrix(basis, wavelengths, filters)

    # Reconstruct coefficients
    coeffs_est = reconstruct_coefficients(phot, design_mat)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectra(coeffs_est, basis)

    # Compare true vs reconstructed spectra (mean absolute error)
    mae = np.mean(np.abs(spectra - recon_spectra), axis=1)
    for i in range(n_samples):
        print(f"Sample {i+1} MAE: {mae[i]:.6f}")

if __name__ == "__main__":
    main()