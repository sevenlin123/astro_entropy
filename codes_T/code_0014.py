import numpy as np
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model
# ----------------------------------------------------------------------
def create_wavelength_grid(start=3000, stop=10000, num=200):
    """Wavelength grid in Angstroms."""
    return np.linspace(start, stop, num)

def create_gaussian_basis(wavelength, centers, widths):
    """
    Create Gaussian basis spectra.
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelength grid.
    centers : list or ndarray
        Centers of the Gaussians.
    widths : list or ndarray
        Standard deviations of the Gaussians.
    
    Returns
    -------
    basis : ndarray
        Shape (num_basis, len(wavelength))
    """
    basis = []
    for c, w in zip(centers, widths):
        gauss = np.exp(-0.5 * ((wavelength - c) / w)**2)
        basis.append(gauss)
    return np.vstack(basis)

# ----------------------------------------------------------------------
# Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(num_samples, basis, coeff_range=(0.1, 1.0)):
    """
    Generate synthetic spectra as random linear combinations of basis.
    
    Parameters
    ----------
    num_samples : int
        Number of spectra to generate.
    basis : ndarray
        Basis spectra (n_basis, n_wave).
    coeff_range : tuple
        Range for random coefficients.
    
    Returns
    -------
    spectra : ndarray
        Shape (num_samples, n_wave)
    coeffs : ndarray
        Shape (num_samples, n_basis)
    """
    n_basis = basis.shape[0]
    coeffs = np.random.uniform(coeff_range[0], coeff_range[1],
                               size=(num_samples, n_basis))
    spectra = coeffs @ basis
    return spectra, coeffs

# ----------------------------------------------------------------------
# Photometric filter generation
# ----------------------------------------------------------------------
def create_filter_passband(wavelength, center, width):
    """
    Gaussian filter passband.
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelength grid.
    center : float
        Center of the filter.
    width : float
        Standard deviation.
    
    Returns
    -------
    transmission : ndarray
    """
    return np.exp(-0.5 * ((wavelength - center) / width)**2)

def generate_filters(wavelength, filter_specs):
    """
    Create multiple filter passbands.
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelength grid.
    filter_specs : list of tuples
        Each tuple is (center, width).
    
    Returns
    -------
    filters : ndarray
        Shape (n_filters, len(wavelength))
    """
    return np.array([create_filter_passband(wavelength, c, w)
                     for c, w in filter_specs])

# ----------------------------------------------------------------------
# Photometry computation
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """
    Compute photometric fluxes by integrating spectrum with filters.
    
    Parameters
    ----------
    spectra : ndarray
        Shape (n_samples, n_wave)
    filters : ndarray
        Shape (n_filters, n_wave)
    
    Returns
    -------
    photometry : ndarray
        Shape (n_samples, n_filters)
    """
    # Simple trapezoidal integration
    return spectra @ filters.T

# ----------------------------------------------------------------------
# Reconstruction from photometry
# ----------------------------------------------------------------------
def reconstruct_spectrum_from_photometry(photometry, filters, basis,
                                          reg_param=0.1):
    """
    Reconstruct spectral coefficients from photometry using ridge regression.
    
    Parameters
    ----------
    photometry : ndarray
        Observed photometric fluxes (n_samples, n_filters).
    filters : ndarray
        Filter passbands (n_filters, n_wave).
    basis : ndarray
        Basis spectra (n_basis, n_wave).
    reg_param : float
        Ridge regularization strength.
    
    Returns
    -------
    reconstructed_spectra : ndarray
        Shape (n_samples, n_wave)
    coeffs_hat : ndarray
        Estimated coefficients (n_samples, n_basis)
    """
    # Build the mapping from coefficients to photometry:
    # phot = coeffs @ (basis * filters.T) => compute effective filter for each basis
    effective_filters = basis @ filters.T  # shape (n_basis, n_filters)
    ridge = Ridge(alpha=reg_param, fit_intercept=False)
    ridge.fit(effective_filters.T, photometry.T)
    coeffs_hat = ridge.coef_.T  # shape (n_samples, n_basis)
    reconstructed_spectra = coeffs_hat @ basis
    return reconstructed_spectra, coeffs_hat

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Define wavelength grid and basis
    wave = create_wavelength_grid()
    centers = [4000, 5000, 6000, 7000, 8000]
    widths  = [200, 200, 200, 200, 200]
    basis = create_gaussian_basis(wave, centers, widths)

    # 2. Generate synthetic spectra
    n_samples = 10
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis)

    # 3. Define photometric filters (UBVRI-like)
    filter_specs = [(3600, 300),   # U
                    (4400, 300),   # B
                    (5500, 300),   # V
                    (6600, 300),   # R
                    (7900, 300)]   # I
    filters = generate_filters(wave, filter_specs)

    # 4. Compute photometry
    phot = compute_photometry(spectra, filters)

    # 5. Reconstruct spectra from photometry
    recon_spectra, est_coeffs = reconstruct_spectrum_from_photometry(
        phot, filters, basis, reg_param=0.01)

    # 6. Compare a sample
    idx = 0
    print("True coefficients:\n", true_coeffs[idx])
    print("Estimated coefficients:\n", est_coeffs[idx])
    print("\nTrue spectrum:\n", spectra[idx])
    print("\nReconstructed spectrum:\n", recon_spectra[idx])