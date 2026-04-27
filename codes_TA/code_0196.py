import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# -----------------------------
# 1. Spectral Model Definition
# -----------------------------

def generate_wavelength_grid(start=400.0, end=800.0, n_points=401):
    """Create a linear wavelength grid (nm)."""
    return np.linspace(start, end, n_points)

def gaussian(x, mu, sigma):
    """Normalized Gaussian function."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_basis_functions(wavelengths, n_basis=10):
    """Generate a set of Gaussian basis functions."""
    mus = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    sigma = (wavelengths.max() - wavelengths.min()) / (2 * n_basis)
    basis = np.array([gaussian(wavelengths, mu, sigma) for mu in mus])
    return basis  # shape: (n_basis, n_wavelengths)

def synthesize_spectrum(coeffs, basis_functions):
    """Compute a synthetic spectrum as a linear combination of basis functions."""
    return coeffs @ basis_functions  # shape: (n_wavelengths,)

# -----------------------------
# 2. Photometric Data Generation
# -----------------------------

def generate_filter_responses(wavelengths, n_filters=3):
    """Generate simple top-hat filter responses."""
    filt_width = (wavelengths.max() - wavelengths.min()) / (n_filters + 1)
    centers = np.linspace(
        wavelengths.min() + filt_width, 
        wavelengths.max() - filt_width, 
        n_filters
    )
    filters = np.zeros((n_filters, len(wavelengths)))
    for i, cen in enumerate(centers):
        left = cen - filt_width / 2
        right = cen + filt_width / 2
        mask = (wavelengths >= left) & (wavelengths <= right)
        filters[i, mask] = 1.0
    return filters  # shape: (n_filters, n_wavelengths)

def compute_photometry(spectrum, filter_responses, wavelengths):
    """Integrate the spectrum through each filter (trapezoidal integration)."""
    dx = np.gradient(wavelengths)
    integrals = []
    for filt in filter_responses:
        integrand = spectrum * filt
        integral = np.sum(integrand * dx)
        integrals.append(integral)
    return np.array(integrals)  # shape: (n_filters,)

# -----------------------------
# 3. Spectrum Reconstruction
# -----------------------------

def prepare_design_matrix(filter_responses, basis_functions, wavelengths):
    """
    Build the design matrix A such that photometry = A @ coeffs.
    Each element A[i,j] = ∫ filter_i(λ) * basis_j(λ) dλ.
    """
    dx = np.gradient(wavelengths)
    A = np.zeros((filter_responses.shape[0], basis_functions.shape[0]))
    for i, filt in enumerate(filter_responses):
        for j, basis in enumerate(basis_functions):
            integrand = filt * basis
            A[i, j] = np.sum(integrand * dx)
    return A

def reconstruct_coefficients(photometry, design_matrix, alpha=0.1):
    """Solve for coefficients using ridge regression."""
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(design_matrix, photometry)
    return reg.coef_  # shape: (n_basis,)

# -----------------------------
# 4. Example Workflow
# -----------------------------

def main():
    # Wavelength grid
    wav = generate_wavelength_grid()

    # Basis functions
    n_basis = 10
    basis = generate_basis_functions(wav, n_basis=n_basis)

    # True coefficients (random)
    rng = np.random.default_rng(seed=42)
    true_coeffs = rng.uniform(0.5, 1.5, size=n_basis)

    # Synthetic spectrum
    true_spec = synthesize_spectrum(true_coeffs, basis)

    # Filter responses
    n_filters = 3
    filters = generate_filter_responses(wav, n_filters=n_filters)

    # Photometric data
    photometry = compute_photometry(true_spec, filters, wav)

    # Design matrix
    A = prepare_design_matrix(filters, basis, wav)

    # Reconstruct coefficients
    recon_coeffs = reconstruct_coefficients(photometry, A, alpha=0.1)

    # Reconstructed spectrum
    recon_spec = synthesize_spectrum(recon_coeffs, basis)

    # Output comparison
    print("True coefficients:\n", true_coeffs)
    print("\nReconstructed coefficients:\n", recon_coeffs)
    print("\nMean absolute error in coefficients:", np.mean(np.abs(true_coeffs - recon_coeffs)))

    # Mean squared error between spectra
    mse_spec = np.mean((true_spec - recon_spec)**2)
    print("MSE between true and reconstructed spectra:", mse_spec)

if __name__ == "__main__":
    main()