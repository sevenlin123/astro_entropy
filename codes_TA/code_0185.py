import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# Spectral model: linear combination of Gaussian basis functions
# ------------------------------------------------------------
def generate_basis_functions(wavelengths, centers, widths):
    """
    Create Gaussian basis functions evaluated at given wavelengths.
    
    Parameters
    ----------
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
    centers : list or ndarray
        Centers of Gaussian basis functions.
    widths : list or ndarray
        Standard deviations of Gaussian basis functions.
        
    Returns
    -------
    basis_funcs : ndarray, shape (N, M)
        Basis functions evaluated on the wavelength grid.
    """
    basis_funcs = []
    for c, w in zip(centers, widths):
        g = np.exp(-0.5 * ((wavelengths - c) / w) ** 2)
        basis_funcs.append(g)
    return np.column_stack(basis_funcs)

# ------------------------------------------------------------
# Synthetic spectra generation
# ------------------------------------------------------------
def generate_synthetic_spectrum(coeffs, basis_funcs):
    """
    Compute a synthetic spectrum as a linear combination of basis functions.
    
    Parameters
    ----------
    coeffs : ndarray, shape (M,)
        Coefficients for each basis function.
    basis_funcs : ndarray, shape (N, M)
        Basis functions.
        
    Returns
    -------
    spectrum : ndarray, shape (N,)
        Synthetic flux values.
    """
    return basis_funcs @ coeffs

# ------------------------------------------------------------
# Photometric filter responses (Gaussian approximations)
# ------------------------------------------------------------
def generate_filter_responses(wavelengths, filter_centers, filter_widths):
    """
    Create Gaussian filter responses evaluated at given wavelengths.
    
    Parameters
    ----------
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
    filter_centers : list or ndarray
        Central wavelengths of the filters.
    filter_widths : list or ndarray
        Standard deviations of the filters.
        
    Returns
    -------
    filters : ndarray, shape (N, K)
        Filter transmission curves.
    """
    filters = []
    for fc, fw in zip(filter_centers, filter_widths):
        filt = np.exp(-0.5 * ((wavelengths - fc) / fw) ** 2)
        filters.append(filt)
    return np.column_stack(filters)

# ------------------------------------------------------------
# Photometry computation
# ------------------------------------------------------------
def compute_photometry(spectrum, filters, wavelengths):
    """
    Integrate spectrum over each filter to obtain photometric fluxes.
    
    Parameters
    ----------
    spectrum : ndarray, shape (N,)
        Spectral flux values.
    filters : ndarray, shape (N, K)
        Filter responses.
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
        
    Returns
    -------
    photometry : ndarray, shape (K,)
        Integrated fluxes per filter.
    """
    # Simple trapezoidal integration over wavelength
    dw = np.diff(wavelengths, prepend=wavelengths[0])
    integrated = np.sum(spectrum[:, None] * filters * dw[:, None], axis=0)
    norm = np.sum(filters * dw[:, None], axis=0)
    return integrated / norm

# ------------------------------------------------------------
# Reconstruction of spectrum coefficients from photometry
# ------------------------------------------------------------
def reconstruct_coefficients(photometry, basis_funcs, filters, wavelengths):
    """
    Estimate basis function coefficients that best reproduce given photometry.
    
    Parameters
    ----------
    photometry : ndarray, shape (K,)
        Observed photometric fluxes.
    basis_funcs : ndarray, shape (N, M)
        Basis functions.
    filters : ndarray, shape (N, K)
        Filter responses.
    wavelengths : ndarray, shape (N,)
        Wavelength grid.
        
    Returns
    -------
    coeffs_est : ndarray, shape (M,)
        Reconstructed coefficients.
    """
    # Build design matrix: each column corresponds to the photometry
    # expected from a unit coefficient in the basis
    dw = np.diff(wavelengths, prepend=wavelengths[0])
    proj = np.zeros((len(photometry), basis_funcs.shape[1]))
    for i in range(basis_funcs.shape[1]):
        proj[:, i] = np.sum(basis_funcs[:, i][:, None] * filters * dw[:, None], axis=0)
    proj /= np.sum(filters * dw[:, None], axis=0)
    
    reg = LinearRegression(fit_intercept=False)
    reg.fit(proj, photometry)
    return reg.coef_

# ------------------------------------------------------------
# Main demonstration
# ------------------------------------------------------------
def main():
    np.random.seed(42)
    
    # Wavelength grid
    wav = np.linspace(300, 800, 501)  # nm
    
    # Basis functions parameters
    basis_centers = [350, 500, 650]
    basis_widths   = [30, 40, 25]
    basis_funcs    = generate_basis_functions(wav, basis_centers, basis_widths)
    
    # Synthetic coefficients
    true_coeffs = np.array([1.0, 0.5, 0.8]) + 0.1 * np.random.randn(3)
    
    # Generate synthetic spectrum
    spec = generate_synthetic_spectrum(true_coeffs, basis_funcs)
    
    # Filters parameters
    filter_centers = [380, 440, 550, 660]  # U, B, V, R
    filter_widths  = [20, 20, 25, 25]
    filters        = generate_filter_responses(wav, filter_centers, filter_widths)
    
    # Compute photometry
    phot = compute_photometry(spec, filters, wav)
    
    # Reconstruct coefficients
    coeffs_rec = reconstruct_coefficients(phot, basis_funcs, filters, wav)
    
    # Reconstruct spectrum
    spec_rec = generate_synthetic_spectrum(coeffs_rec, basis_funcs)
    
    # Print results
    print("True coefficients :", true_coeffs)
    print("Reconstructed coeffs:", coeffs_rec)
    print("\nMean squared error between spectra:", np.mean((spec - spec_rec)**2))
    
if __name__ == "__main__":
    main()