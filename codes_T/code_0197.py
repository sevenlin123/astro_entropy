import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----------------------------
# Spectral model definition
# ----------------------------

def gaussian_basis(wave, centers, widths):
    """Return a (N_basis x N_wave) matrix of Gaussian basis functions."""
    return np.exp(-0.5 * ((wave[:, None] - centers[None, :]) / widths[None, :])**2)

def spectral_model(coeffs, wave):
    """
    Generate a synthetic spectrum as a linear combination of Gaussian bases.
    
    Parameters
    ----------
    coeffs : array_like, shape (N_basis,)
        Amplitudes of the basis functions.
    wave   : array_like, shape (N_wave,)
        Wavelength grid.
    
    Returns
    -------
    spectrum : ndarray, shape (N_wave,)
        Resulting synthetic spectrum.
    """
    bases = gaussian_basis(wave, BASIS_CENTERS, BASIS_WIDTHS)
    return bases @ coeffs

# ----------------------------
# Filter generation
# ----------------------------

def gaussian_filter(wave, center, width):
    """Single Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wave - center)/width)**2)

def generate_filters(wave):
    """Return a list of filter transmission curves."""
    filt_centers = [4500., 5500., 6500.]
    filt_widths  = [300., 300., 300.]
    return [gaussian_filter(wave, c, w) for c, w in zip(filt_centers, filt_widths)]

# ----------------------------
# Synthetic data generation
# ----------------------------

def generate_synthetic_spectra(n_samples, wave):
    """Create synthetic spectra by sampling random coefficients."""
    coeffs = np.random.randn(n_samples, N_BASIS)
    spectra = np.array([spectral_model(c, wave) for c in coeffs])
    return coeffs, spectra

def compute_photometry(spectra, filters, wave):
    """
    Integrate spectra over filter transmission curves.
    
    Parameters
    ----------
    spectra : ndarray, shape (N_samples, N_wave)
    filters : list of arrays, each shape (N_wave,)
    wave    : array_like, shape (N_wave,)
    
    Returns
    -------
    phot : ndarray, shape (N_samples, N_filters)
    """
    phot = []
    for filt in filters:
        flux = simps(spectra * filt[None, :], wave, axis=1)
        phot.append(flux)
    return np.stack(phot, axis=-1)

# ----------------------------
# Reconstruction
# ----------------------------

def build_design_matrix(filters, wave):
    """
    Build the linear mapping from basis coefficients to photometric fluxes.
    
    M_ij = integral of (basis_j * filter_i) over wavelength.
    """
    bases = gaussian_basis(wave, BASIS_CENTERS, BASIS_WIDTHS)
    M = np.zeros((len(filters), N_BASIS))
    for i, filt in enumerate(filters):
        for j in range(N_BASIS):
            M[i, j] = simps(bases[:, j] * filt, wave)
    return M

def reconstruct_spectrum(photometry, M, wave, alpha=1e-2):
    """
    Reconstruct spectrum from photometry by solving for basis coefficients.
    
    Uses ridge regression to handle ill‑conditioned systems.
    """
    # Fit ridge regression model
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(M, photometry.T)            # transpose so shape matches (n_samples, n_features)
    coeffs_pred = reg.coef_.T            # shape (n_samples, N_BASIS)
    spectra_rec = np.array([spectral_model(c, wave) for c in coeffs_pred])
    return coeffs_pred, spectra_rec

# ----------------------------
# Main execution
# ----------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # Define wavelength grid
    WAVE_MIN, WAVE_MAX, N_WAVE = 4000., 8000., 1000
    WAVELENGTH = np.linspace(WAVE_MIN, WAVE_MAX, N_WAVE)

    # Basis configuration
    N_BASIS = 5
    BASIS_CENTERS = np.linspace(4200., 7800., N_BASIS)
    BASIS_WIDTHS  = np.full(N_BASIS, 300.)

    # Generate synthetic data
    N_SAMPLES = 50
    true_coeffs, spectra_true = generate_synthetic_spectra(N_SAMPLES, WAVELENGTH)

    # Filters
    filters = generate_filters(WAVELENGTH)

    # Photometry
    phot = compute_photometry(spectra_true, filters, WAVELENGTH)

    # Build design matrix
    M = build_design_matrix(filters, WAVELENGTH)

    # Reconstruction
    coeffs_rec, spectra_rec = reconstruct_spectrum(phot, M, WAVELENGTH)

    # Simple validation: print RMSE between true and reconstructed spectra
    rmse = np.sqrt(np.mean((spectra_true - spectra_rec)**2, axis=1))
    print("RMSE per spectrum:", rmse[:5], "...")
    print("Average RMSE:", rmse.mean())