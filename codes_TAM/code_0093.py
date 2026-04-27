import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
#                     Spectral model (basis set)                              #
# --------------------------------------------------------------------------- #

def gaussian_basis(n_basis, wavelengths):
    """
    Construct a set of Gaussian basis spectra.
    
    Parameters
    ----------
    n_basis : int
        Number of basis spectra.
    wavelengths : ndarray
        Array of wavelength values (in nm).
        
    Returns
    -------
    basis : ndarray
        Shape (n_basis, len(wavelengths))
    """
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths =  (wavelengths.max() - wavelengths.min()) / (2 * n_basis)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths)**2)
    return basis

# --------------------------------------------------------------------------- #
#           Synthetic spectra generation from random coefficients            #
# --------------------------------------------------------------------------- #

def generate_synthetic_spectra(n_samples, basis, rng=None):
    """
    Generate synthetic spectra as linear combinations of basis spectra.
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic spectra to generate.
    basis : ndarray
        Basis spectra matrix (n_basis, n_wavelengths).
    rng : np.random.Generator, optional
        Random number generator instance.
        
    Returns
    -------
    spectra : ndarray
        Shape (n_samples, n_wavelengths)
    coeffs : ndarray
        Shape (n_samples, n_basis)
    """
    rng = rng or np.random.default_rng()
    coeffs = rng.uniform(0.0, 1.0, size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis
    return spectra, coeffs

# --------------------------------------------------------------------------- #
#           Photometric data simulation (filter integration)                 #
# --------------------------------------------------------------------------- #

def photometric_fluxes(spectra, filter_transmissions, wavelengths):
    """
    Compute photometric fluxes by integrating spectra over filter curves.
    
    Parameters
    ----------
    spectra : ndarray
        Shape (n_samples, n_wavelengths)
    filter_transmissions : ndarray
        Shape (n_filters, n_wavelengths)
    wavelengths : ndarray
        Wavelength array corresponding to spectra and filters.
        
    Returns
    -------
    fluxes : ndarray
        Shape (n_samples, n_filters)
    """
    # Ensure arrays are numpy
    spectra = np.asarray(spectra)
    filter_transmissions = np.asarray(filter_transmissions)
    # Element‑wise multiplication then trapezoidal integration
    integrands = spectra[:, :, None] * filter_transmissions[None, :, :]
    fluxes = np.trapz(integrands, wavelengths, axis=1)
    return fluxes

# --------------------------------------------------------------------------- #
#          Reconstruction of spectra from photometric measurements           #
# --------------------------------------------------------------------------- #

def reconstruct_spectra_from_photometry(fluxes, filter_transmissions,
                                        wavelengths, basis, alpha=1.0):
    """
    Reconstruct spectra by solving a regularised least‑squares problem.
    
    Parameters
    ----------
    fluxes : ndarray
        Shape (n_samples, n_filters)
    filter_transmissions : ndarray
        Shape (n_filters, n_wavelengths)
    wavelengths : ndarray
        Wavelength array.
    basis : ndarray
        Basis spectra matrix (n_basis, n_wavelengths).
    alpha : float, optional
        Regularisation strength for Ridge regression.
        
    Returns
    -------
    reconstructed_spectra : ndarray
        Shape (n_samples, n_wavelengths)
    coeffs_pred : ndarray
        Predicted coefficients (n_samples, n_basis)
    """
    # Build design matrix: filter response integrated against basis
    design = np.trapz(basis.T * filter_transmissions[:, None, :], wavelengths, axis=2)
    # Fit ridge regression to predict fluxes from coefficients
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(design.T, fluxes.T)
    coeffs_pred = reg.coef_.T
    reconstructed_spectra = coeffs_pred @ basis
    return reconstructed_spectra, coeffs_pred

# --------------------------------------------------------------------------- #
#                          Example usage                                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Define wavelength grid (400–800 nm)
    wav = np.linspace(400.0, 800.0, 500)

    # Create basis spectra
    n_basis = 10
    basis = gaussian_basis(n_basis, wav)

    # Generate synthetic spectra
    n_spectra = 20
    spectra, true_coeffs = generate_synthetic_spectra(n_spectra, basis)

    # Define three mock photometric filters (Gaussian passbands)
    n_filters = 3
    filt_centers = np.array([450, 600, 750])
    filt_width = 50.0
    filters = np.exp(-0.5 * ((wav[:, None] - filt_centers[None, :]) / filt_width)**2)

    # Simulate photometric fluxes
    fluxes = photometric_fluxes(spectra, filters, wav)

    # Reconstruct spectra from photometry
    recon_spectra, pred_coeffs = reconstruct_spectra_from_photometry(
        fluxes, filters, wav, basis, alpha=0.01
    )

    # Simple verification: compare true vs predicted coefficients
    print("True coefficients vs. Predicted coefficients (first 3 samples):")
    print(np.hstack((true_coeffs[:3], pred_coeffs[:3])))