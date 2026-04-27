import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model
# ----------------------------------------------------------------------
def gaussian(x, center, sigma):
    """One‑dimensional Gaussian."""
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)

def build_basis(wavelength, centers, sigma=10.0):
    """
    Construct a set of Gaussian basis functions.
    
    Parameters
    ----------
    wavelength : array_like
        Wavelength grid (in nm).
    centers : array_like
        Centers of the Gaussians (in nm).
    sigma : float
        Width of each Gaussian (in nm).
    
    Returns
    -------
    basis : ndarray, shape (n_basis, len(wavelength))
        Array of basis functions evaluated on the wavelength grid.
    """
    basis = np.vstack([gaussian(wavelength, c, sigma) for c in centers])
    return basis

def synthesize_spectrum(coeffs, basis):
    """
    Combine basis functions with coefficients to obtain a spectrum.
    
    Parameters
    ----------
    coeffs : array_like, shape (n_basis,)
        Coefficients of the basis functions.
    basis : ndarray, shape (n_basis, n_wave)
        Basis functions.
    
    Returns
    -------
    flux : ndarray, shape (n_wave,)
        Synthetic spectrum.
    """
    return np.dot(coeffs, basis)

# ----------------------------------------------------------------------
# Photometry generation
# ----------------------------------------------------------------------
def build_filters(wavelength, filter_edges):
    """
    Build simple top‑hat filter transmission curves.
    
    Parameters
    ----------
    wavelength : array_like
        Wavelength grid (in nm).
    filter_edges : list of tuples
        Each tuple contains (λ_min, λ_max) for a filter.
    
    Returns
    -------
    filters : ndarray, shape (n_filters, len(wavelength))
        Filter transmission curves (1 inside the interval, 0 outside).
    """
    filters = []
    for lam_min, lam_max in filter_edges:
        filt = np.logical_and(wavelength >= lam_min, wavelength <= lam_max).astype(float)
        filters.append(filt)
    return np.vstack(filters)

def compute_photometry(spectrum, filters):
    """
    Compute integrated photometric fluxes for a given spectrum.
    
    Parameters
    ----------
    spectrum : array_like, shape (n_wave,)
        Spectrum to integrate.
    filters : ndarray, shape (n_filters, n_wave)
        Filter transmission curves.
    
    Returns
    -------
    phot : ndarray, shape (n_filters,)
        Integrated fluxes in each filter.
    """
    return np.array([simps(spectrum * f, x=wavelength) for f in filters])

# ----------------------------------------------------------------------
# Reconstruction framework
# ----------------------------------------------------------------------
def build_response_matrix(basis, filters, wavelength):
    """
    Build the linear mapping from basis coefficients to photometric fluxes.
    
    Parameters
    ----------
    basis : ndarray, shape (n_basis, n_wave)
        Basis functions.
    filters : ndarray, shape (n_filters, n_wave)
        Filter transmissions.
    wavelength : array_like
        Wavelength grid.
    
    Returns
    -------
    A : ndarray, shape (n_filters, n_basis)
        Response matrix: A[j,i] = ∫ basis_i(λ) * filter_j(λ) dλ
    """
    n_filters, _ = filters.shape
    n_basis, _ = basis.shape
    A = np.zeros((n_filters, n_basis))
    for j in range(n_filters):
        for i in range(n_basis):
            A[j, i] = simps(basis[i] * filters[j], x=wavelength)
    return A

def reconstruct_coefficients(photometry, A, alpha=1.0):
    """
    Recover basis coefficients by ridge regression.
    
    Parameters
    ----------
    photometry : array_like, shape (n_filters,)
        Observed photometric fluxes.
    A : ndarray, shape (n_filters, n_basis)
        Response matrix.
    alpha : float
        Regularization strength for ridge regression.
    
    Returns
    -------
    coeffs : ndarray, shape (n_basis,)
        Reconstructed coefficients.
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(A, photometry)
    return ridge.coef_

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wavelength = np.arange(400, 801, 1)  # 400–800 nm
    
    # Define basis
    centers = np.array([450, 500, 550, 600, 650])  # nm
    basis = build_basis(wavelength, centers, sigma=15.0)
    
    # Define filters (top‑hat)
    filter_edges = [(420, 460), (480, 520), (580, 620)]
    filters = build_filters(wavelength, filter_edges)
    
    # Build response matrix
    A = build_response_matrix(basis, filters, wavelength)
    
    # Generate synthetic spectra
    n_samples = 10
    rng = np.random.default_rng(seed=42)
    coeffs_true = rng.normal(size=(n_samples, len(centers)))   # true coefficients
    spectra = np.array([synthesize_spectrum(c, basis) for c in coeffs_true])
    
    # Compute photometry
    phot = np.array([compute_photometry(s, filters) for s in spectra])
    
    # Reconstruct first spectrum
    idx = 0
    coeffs_rec = reconstruct_coefficients(phot[idx], A, alpha=0.1)
    spectrum_rec = synthesize_spectrum(coeffs_rec, basis)
    
    # Print results
    print("True coefficients:\n", coeffs_true[idx])
    print("\nReconstructed coefficients:\n", coeffs_rec)
    print("\nReconstruction error (L2 norm):", np.linalg.norm(coeffs_true[idx]-coeffs_rec))