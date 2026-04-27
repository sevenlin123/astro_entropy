import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ---------------------------------------------
# 1. Spectral model – basis functions
# ---------------------------------------------
def generate_basis(n_bases, n_wl, wl_min=300.0, wl_max=800.0, seed=42):
    """
    Create a set of Gaussian basis spectra.
    
    Parameters
    ----------
    n_bases : int
        Number of basis functions.
    n_wl   : int
        Number of wavelength points.
    wl_min : float
        Minimum wavelength (nm).
    wl_max : float
        Maximum wavelength (nm).
    seed   : int
        Random seed for reproducibility.
    
    Returns
    -------
    wavelengths : ndarray shape (n_wl,)
        Wavelength grid.
    basis       : ndarray shape (n_wl, n_bases)
        Basis spectra.
    """
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(wl_min, wl_max, n_wl)

    # Random centers and widths
    centers = rng.uniform(wl_min, wl_max, size=n_bases)
    widths  = rng.uniform((wl_max-wl_min)/10, (wl_max-wl_min)/4, size=n_bases)

    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers)**2) / widths**2)
    # Normalize each basis
    basis /= np.max(basis, axis=0)
    return wavelengths, basis


# ---------------------------------------------
# 2. Generate synthetic spectra
# ---------------------------------------------
def generate_synthetic_spectrum(basis, coeffs=None, noise_level=0.01, seed=24):
    """
    Construct a synthetic spectrum as a linear combination of basis spectra.
    
    Parameters
    ----------
    basis      : ndarray shape (n_wl, n_bases)
        Basis spectra.
    coeffs     : ndarray shape (n_bases,) or None
        Coefficients for the linear combination. If None, random coefficients
        are drawn uniformly between 0 and 1.
    noise_level: float
        Relative noise amplitude to add to the spectrum.
    seed       : int
        Random seed for reproducibility.
    
    Returns
    -------
    spectrum   : ndarray shape (n_wl,)
        Synthetic spectrum.
    coeffs     : ndarray shape (n_bases,)
        Coefficients used.
    """
    rng = np.random.default_rng(seed)
    n_bases = basis.shape[1]
    if coeffs is None:
        coeffs = rng.uniform(0.0, 1.0, size=n_bases)
    spectrum = basis @ coeffs
    # Add Gaussian noise
    noise = rng.normal(scale=noise_level * np.max(spectrum))
    spectrum += noise
    return spectrum, coeffs


# ---------------------------------------------
# 3. Photometry generation
# ---------------------------------------------
def make_filter_response(wavelengths, centre, width, shape='gaussian'):
    """
    Create a simple filter response curve.
    
    Parameters
    ----------
    wavelengths : ndarray
        Wavelength grid.
    centre      : float
        Central wavelength of the filter.
    width       : float
        Width (FWHM) of the filter.
    shape       : str
        Shape of the filter ('gaussian' or 'rectangular').
    
    Returns
    -------
    response    : ndarray
        Filter response vector.
    """
    if shape == 'gaussian':
        sigma = width / 2.3548  # FWHM -> sigma
        response = np.exp(-0.5 * ((wavelengths - centre) / sigma)**2)
    elif shape == 'rectangular':
        response = np.logical_and(
            wavelengths >= centre - width/2,
            wavelengths <= centre + width/2).astype(float)
    else:
        raise ValueError(f"Unknown shape {shape}")
    # Normalize to unit integral
    response /= simps(response, wavelengths)
    return response


def compute_photometry(spectrum, wavelengths, filters):
    """
    Compute synthetic photometric fluxes by integrating the spectrum through each filter.
    
    Parameters
    ----------
    spectrum     : ndarray shape (n_wl,)
        Spectrum to be integrated.
    wavelengths  : ndarray shape (n_wl,)
        Wavelength grid.
    filters      : list of ndarray
        List of filter response vectors.
    
    Returns
    -------
    photometry   : ndarray shape (n_filters,)
        Integrated fluxes in each filter.
    """
    photometry = np.array([simps(spectrum * f, wavelengths) for f in filters])
    return photometry


# ---------------------------------------------
# 4. Reconstruction
# ---------------------------------------------
def reconstruct_spectrum(photometry, filters, basis, wavelengths):
    """
    Estimate the coefficients of the basis spectra that best reproduce the observed photometry
    and reconstruct the full spectrum.
    
    Parameters
    ----------
    photometry  : ndarray shape (n_filters,)
        Observed photometric fluxes.
    filters     : list of ndarray
        Filter response vectors (each of length n_wl).
    basis       : ndarray shape (n_wl, n_bases)
        Basis spectra.
    wavelengths : ndarray shape (n_wl,)
        Wavelength grid.
    
    Returns
    -------
    recon_spectrum : ndarray shape (n_wl,)
        Reconstructed spectrum.
    coeffs         : ndarray shape (n_bases,)
        Estimated coefficients.
    """
    # Build matrix that maps basis coefficients to photometry
    # Each column corresponds to a basis; each row to a filter
    filt_mat = np.vstack([simps(basis * f[:, None], wavelengths) for f in filters])
    # Solve least squares problem
    lr = LinearRegression(fit_intercept=False, positive=True)
    lr.fit(filt_mat.T, photometry)
    coeffs = lr.coef_
    recon_spectrum = basis @ coeffs
    return recon_spectrum, coeffs


# ---------------------------------------------
# 5. Example usage
# ---------------------------------------------
def main():
    # Setup wavelength grid
    n_wl = 1000
    wavelengths, basis = generate_basis(n_bases=5, n_wl=n_wl, seed=1)
    
    # Create three mock filters (U, B, V)
    filt_U = make_filter_response(wavelengths, centre=360, width=50)
    filt_B = make_filter_response(wavelengths, centre=440, width=50)
    filt_V = make_filter_response(wavelengths, centre=550, width=70)
    filters = [filt_U, filt_B, filt_V]
    
    # Generate a synthetic spectrum
    true_spectrum, true_coeffs = generate_synthetic_spectrum(basis, noise_level=0.02, seed=2)
    
    # Compute synthetic photometry
    photometry = compute_photometry(true_spectrum, wavelengths, filters)
    
    # Reconstruct spectrum from photometry
    recon_spectrum, est_coeffs = reconstruct_spectrum(photometry, filters, basis, wavelengths)
    
    # Print results
    print("True coefficients :", true_coeffs)
    print("Estimated coeffs :", est_coeffs)
    # Optionally, evaluate reconstruction error
    err = np.linalg.norm(true_spectrum - recon_spectrum) / np.linalg.norm(true_spectrum)
    print(f"Relative reconstruction error: {err:.4f}")


if __name__ == "__main__":
    main()