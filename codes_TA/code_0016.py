import numpy as np
from sklearn.linear_model import Ridge
from scipy.signal import gaussian

# --------------------------------------------------------------------
# 1. Define spectral model (basis functions)
# --------------------------------------------------------------------
def generate_basis(n_basis, n_points):
    """
    Generate a set of orthogonal basis spectra.
    
    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    n_points : int
        Number of wavelength points per spectrum.
    
    Returns
    -------
    basis : ndarray, shape (n_basis, n_points)
        Basis spectra.
    """
    wavelengths = np.linspace(0, 1, n_points)
    basis = []
    for k in range(n_basis):
        # Simple sinusoidal basis
        basis.append(np.sin((k + 1) * np.pi * wavelengths))
    return np.array(basis)

# --------------------------------------------------------------------
# 2. Generate synthetic spectra
# --------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis, coeff_min=-1, coeff_max=1,
                               noise_std=0.05):
    """
    Generate synthetic spectra as linear combinations of basis spectra.
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic spectra to generate.
    basis : ndarray, shape (n_basis, n_points)
        Basis spectra.
    coeff_min, coeff_max : float
        Range of coefficients for linear combination.
    noise_std : float
        Standard deviation of Gaussian noise added to spectra.
    
    Returns
    -------
    spectra : ndarray, shape (n_samples, n_points)
        Synthetic spectra.
    coeffs : ndarray, shape (n_samples, n_basis)
        Coefficients used to generate each spectrum.
    """
    n_basis, n_points = basis.shape
    coeffs = np.random.uniform(coeff_min, coeff_max,
                               size=(n_samples, n_basis))
    spectra = coeffs @ basis  # matrix multiplication
    noise = np.random.normal(scale=noise_std, size=spectra.shape)
    spectra += noise
    return spectra, coeffs

# --------------------------------------------------------------------
# 3. Generate photometric filter responses
# --------------------------------------------------------------------
def generate_filters(n_filters, n_points, width=0.1):
    """
    Create Gaussian filter response curves.
    
    Parameters
    ----------
    n_filters : int
        Number of filters.
    n_points : int
        Number of wavelength points per filter.
    width : float
        Standard deviation of Gaussian filters.
    
    Returns
    -------
    filters : ndarray, shape (n_filters, n_points)
        Filter transmission curves.
    """
    wavelengths = np.linspace(0, 1, n_points)
    centers = np.linspace(0.1, 0.9, n_filters)
    filters = []
    for c in centers:
        filt = np.exp(-0.5 * ((wavelengths - c) / width)**2)
        filt /= filt.sum()
        filters.append(filt)
    return np.array(filters)

# --------------------------------------------------------------------
# 4. Compute photometric fluxes from spectra
# --------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """
    Integrate spectra over filter transmission curves to get fluxes.
    
    Parameters
    ----------
    spectra : ndarray, shape (n_samples, n_points)
        Spectra.
    filters : ndarray, shape (n_filters, n_points)
        Filter responses.
    
    Returns
    -------
    photometry : ndarray, shape (n_samples, n_filters)
        Fluxes in each filter.
    """
    return spectra @ filters.T  # matrix multiplication

# --------------------------------------------------------------------
# 5. Reconstruct spectra from photometry
# --------------------------------------------------------------------
def reconstruct_spectra(photometry, filters, basis, alpha=0.01):
    """
    Reconstruct spectra by solving a linear system:
    
        photometry ≈ reconstructed_spectrum · filtersᵀ
    
    We first solve for the linear combination of basis functions that
    best reproduces the photometry, then form the spectrum.
    
    Parameters
    ----------
    photometry : ndarray, shape (n_samples, n_filters)
        Observed fluxes.
    filters : ndarray, shape (n_filters, n_points)
        Filter responses.
    basis : ndarray, shape (n_basis, n_points)
        Basis spectra.
    alpha : float
        Regularization strength for ridge regression.
    
    Returns
    -------
    reconstructed : ndarray, shape (n_samples, n_points)
        Reconstructed spectra.
    """
    # Build design matrix: filter responses projected onto basis
    # For each filter, we compute its projection on each basis function
    n_filters, n_points = filters.shape
    n_basis = basis.shape[0]
    design = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            design[i, j] = np.dot(filters[i], basis[j])
    # Fit ridge regression: coeffs = (designᵀ design + αI)^(-1) designᵀ photometryᵀ
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(design, photometry.T)
    coeffs = ridge.coef_.T  # shape (n_samples, n_basis)
    # Reconstruct spectra
    reconstructed = coeffs @ basis
    return reconstructed

# --------------------------------------------------------------------
# 6. Main routine
# --------------------------------------------------------------------
def main():
    np.random.seed(42)
    n_samples = 200   # number of synthetic stars
    n_basis = 5       # number of basis spectra
    n_points = 500    # spectral resolution
    n_filters = 4     # number of photometric bands
    
    # Generate basis spectra
    basis = generate_basis(n_basis, n_points)
    
    # Generate synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(
        n_samples, basis, coeff_min=-1, coeff_max=1, noise_std=0.02)
    
    # Generate filter responses
    filters = generate_filters(n_filters, n_points, width=0.07)
    
    # Compute photometric fluxes
    photometry = compute_photometry(spectra, filters)
    
    # Reconstruct spectra from photometry
    reconstructed_spectra = reconstruct_spectra(photometry, filters, basis,
                                                alpha=0.1)
    
    # Evaluate reconstruction error (mean squared error)
    mse = np.mean((spectra - reconstructed_spectra)**2)
    print(f"Mean Squared Error of reconstruction: {mse:.6f}")
    
    # Show a sample comparison
    idx = 0
    print("\nSample spectrum vs. reconstruction:")
    print("Wavelength indices:", np.arange(n_points))
    print("True spectrum   :", spectra[idx][:10])
    print("Reconstructed   :", reconstructed_spectra[idx][:10])

if __name__ == "__main__":
    main()