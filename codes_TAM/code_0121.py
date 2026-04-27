import numpy as np
from scipy.optimize import least_squares

# ------------------------------------
# Spectral model – Gaussian basis
# ------------------------------------
def gaussian(x, mu, sigma):
    """Gaussian profile."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_basis_functions(wavelengths, params):
    """
    Create a list of Gaussian basis functions.
    
    Parameters
    ----------
    wavelengths : ndarray
        Array of wavelength points.
    params : list of tuples
        Each tuple contains (center, sigma) for a Gaussian.
        
    Returns
    -------
    basis_funcs : list of ndarray
        Basis functions evaluated on `wavelengths`.
    """
    return [gaussian(wavelengths, mu, sigma) for mu, sigma in params]

def synth_spectrum(basis_funcs, coeffs):
    """
    Construct a synthetic spectrum as a linear combination of basis functions.
    
    Parameters
    ----------
    basis_funcs : list of ndarray
        Basis functions.
    coeffs : ndarray
        Coefficients for each basis function.
        
    Returns
    -------
    spectrum : ndarray
        Synthetic spectrum.
    """
    return sum(c * f for c, f in zip(coeffs, basis_funcs))

# ------------------------------------
# Photometric system
# ------------------------------------
def top_hat_filter(wavelengths, center, width):
    """
    Simple top‑hat transmission curve.
    
    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array.
    center : float
        Center wavelength of the filter.
    width : float
        Full width at half maximum (approx.) of the filter.
        
    Returns
    -------
    transmission : ndarray
        Filter transmission (1 inside band, 0 outside).
    """
    half_width = width / 2.0
    return np.where(np.abs(wavelengths - center) <= half_width, 1.0, 0.0)

def generate_filters(wavelengths, centers, widths):
    """
    Generate a set of top‑hat filters.
    
    Parameters
    ----------
    wavelengths : ndarray
        Wavelength array.
    centers : list of float
        Filter center wavelengths.
    widths : list of float
        Filter widths.
        
    Returns
    -------
    filters : list of ndarray
        Transmission curves for each filter.
    """
    return [top_hat_filter(wavelengths, c, w) for c, w in zip(centers, widths)]

def compute_flux(spectrum, filt, wavelengths):
    """
    Compute the photometric flux by integrating spectrum × filter.
    
    Parameters
    ----------
    spectrum : ndarray
        Spectrum array.
    filt : ndarray
        Filter transmission curve.
    wavelengths : ndarray
        Wavelength array.
        
    Returns
    -------
    flux : float
        Integrated flux.
    """
    return np.trapz(spectrum * filt, wavelengths)

def generate_photometry(spectrum, filters, wavelengths):
    """
    Generate fluxes for all filters.
    
    Parameters
    ----------
    spectrum : ndarray
        Input spectrum.
    filters : list of ndarray
        Filter transmissions.
    wavelengths : ndarray
        Wavelength array.
        
    Returns
    -------
    fluxes : ndarray
        Flux in each filter.
    """
    return np.array([compute_flux(spectrum, filt, wavelengths) for filt in filters])

# ------------------------------------
# Reconstruction
# ------------------------------------
def build_design_matrix(basis_funcs, filters, wavelengths):
    """
    Build the matrix relating basis coefficients to filter fluxes.
    
    Parameters
    ----------
    basis_funcs : list of ndarray
        Basis functions.
    filters : list of ndarray
        Filter transmissions.
    wavelengths : ndarray
        Wavelength array.
        
    Returns
    -------
    M : ndarray
        Design matrix of shape (n_filters, n_basis).
    """
    n_filters = len(filters)
    n_basis = len(basis_funcs)
    M = np.empty((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j, bf in enumerate(basis_funcs):
            M[i, j] = np.trapz(bf * filt, wavelengths)
    return M

def reconstruct_coeffs(fluxes, basis_funcs, filters, wavelengths):
    """
    Solve for the best‑fit coefficients given observed photometry.
    
    Parameters
    ----------
    fluxes : ndarray
        Observed filter fluxes.
    basis_funcs : list of ndarray
        Basis functions.
    filters : list of ndarray
        Filter transmissions.
    wavelengths : ndarray
        Wavelength array.
        
    Returns
    -------
    coeffs : ndarray
        Estimated coefficients.
    """
    M = build_design_matrix(basis_funcs, filters, wavelengths)
    # Least‑squares solution
    coeffs, *_ = np.linalg.lstsq(M, fluxes, rcond=None)
    return coeffs

# ------------------------------------
# Main demonstration
# ------------------------------------
def main():
    np.random.seed(42)
    
    # Wavelength grid
    wavelengths = np.linspace(300.0, 1200.0, 901)  # 1 nm resolution
    
    # Define basis functions (Gaussians)
    basis_params = [(500.0, 30.0), (750.0, 30.0), (1000.0, 30.0)]
    basis_funcs = generate_basis_functions(wavelengths, basis_params)
    
    # Random coefficients for synthetic spectrum
    true_coeffs = np.random.uniform(0.5, 2.0, size=len(basis_funcs))
    true_spectrum = synth_spectrum(basis_funcs, true_coeffs)
    
    # Define filters
    filter_centers = [400.0, 550.0, 700.0, 850.0, 1000.0]
    filter_widths = [80.0, 80.0, 80.0, 80.0, 80.0]
    filters = generate_filters(wavelengths, filter_centers, filter_widths)
    
    # Generate photometric observations
    observed_fluxes = generate_photometry(true_spectrum, filters, wavelengths)
    
    # Reconstruct spectrum
    estimated_coeffs = reconstruct_coeffs(observed_fluxes, basis_funcs, filters, wavelengths)
    reconstructed_spectrum = synth_spectrum(basis_funcs, estimated_coeffs)
    
    # Evaluate
    mse = np.mean((true_spectrum - reconstructed_spectrum) ** 2)
    print(f"Mean squared error of reconstruction: {mse:.4e}")
    print("True coefficients:", true_coeffs)
    print("Estimated coefficients:", estimated_coeffs)

if __name__ == "__main__":
    main()