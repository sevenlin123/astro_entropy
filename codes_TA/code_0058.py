import numpy as np
from sklearn.linear_model import Ridge
from scipy.integrate import simps

# ------------------------------------------------------------
# Basis function construction
# ------------------------------------------------------------
def gaussian_basis(wavelengths, centers, widths):
    """
    Create a set of Gaussian basis functions.
    
    Parameters
    ----------
    wavelengths : ndarray
        Wavelength grid.
    centers : list or ndarray
        Center wavelengths of the Gaussians.
    widths : float or list/ndarray
        Standard deviations of the Gaussians.
        
    Returns
    -------
    basis : ndarray
        Array of shape (n_basis, len(wavelengths)).
    """
    if np.isscalar(widths):
        widths = [widths] * len(centers)
    basis = []
    for c, w in zip(centers, widths):
        g = np.exp(-0.5 * ((wavelengths - c) / w) ** 2)
        basis.append(g)
    return np.vstack(basis)


# ------------------------------------------------------------
# Synthetic spectra generation
# ------------------------------------------------------------
def generate_synthetic_spectra(n_objects, basis, rng=None):
    """
    Generate synthetic spectra as linear combinations of basis functions.
    
    Parameters
    ----------
    n_objects : int
        Number of synthetic objects.
    basis : ndarray
        Basis functions (n_basis, n_wavelengths).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
        
    Returns
    -------
    spectra : ndarray
        Generated spectra (n_objects, n_wavelengths).
    weights : ndarray
        Coefficients used to generate spectra (n_objects, n_basis).
    """
    rng = np.random.default_rng(rng)
    n_basis = basis.shape[0]
    weights = rng.standard_normal((n_objects, n_basis))
    spectra = weights @ basis
    return spectra, weights


# ------------------------------------------------------------
# Filter definitions
# ------------------------------------------------------------
def create_tophat_filter(wavelengths, low, high):
    """Create a top-hat filter transmission curve."""
    return ((wavelengths >= low) & (wavelengths <= high)).astype(float)


# ------------------------------------------------------------
# Photometry calculation
# ------------------------------------------------------------
def compute_photometry(spectra, wavelengths, filters):
    """
    Compute synthetic photometry by integrating spectra through filters.
    
    Parameters
    ----------
    spectra : ndarray
        Spectra (n_objects, n_wavelengths).
    wavelengths : ndarray
        Wavelength grid.
    filters : list of tuples
        Each tuple is (name, transmission_curve).
        
    Returns
    -------
    phot : ndarray
        Photometric fluxes (n_objects, n_filters).
    """
    phot = []
    for name, filt in filters:
        # Integrate flux * filter; normalize by filter width
        integrand = spectra * filt
        flux = simps(integrand, wavelengths, axis=1)
        norm = simps(filt, wavelengths)
        phot.append(flux / norm)
    return np.vstack(phot).T


# ------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------
def reconstruct_weights(photometry, basis, wavelengths, filters, alpha=0.1):
    """
    Reconstruct basis coefficients from photometry using ridge regression.
    
    Parameters
    ----------
    photometry : ndarray
        Observed photometric fluxes (n_objects, n_filters).
    basis : ndarray
        Basis functions (n_basis, n_wavelengths).
    wavelengths : ndarray
        Wavelength grid.
    filters : list of tuples
        Filter definitions.
    alpha : float
        Regularization strength for Ridge.
        
    Returns
    -------
    coeffs : ndarray
        Reconstructed coefficients (n_objects, n_basis).
    """
    n_basis = basis.shape[0]
    # Build design matrix: for each filter and basis, integrate basis * filter
    design = []
    for name, filt in filters:
        row = []
        for i in range(n_basis):
            integrand = basis[i] * filt
            val = simps(integrand, wavelengths)
            row.append(val)
        design.append(row)
    design = np.array(design).T  # shape (n_filters, n_basis)
    
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(design, photometry.T)
    coeffs = ridge.predict(design).T
    return coeffs


def reconstruct_spectra(coeffs, basis):
    """
    Reconstruct full spectra from basis coefficients.
    
    Parameters
    ----------
    coeffs : ndarray
        Coefficients (n_objects, n_basis).
    basis : ndarray
        Basis functions (n_basis, n_wavelengths).
        
    Returns
    -------
    spectra : ndarray
        Reconstructed spectra (n_objects, n_wavelengths).
    """
    return coeffs @ basis


# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def main():
    # Wavelength grid
    wl = np.linspace(400, 800, 500)  # nm
    
    # Basis functions: 5 Gaussians centered evenly between 420 and 780 nm
    centers = np.linspace(420, 780, 5)
    widths = 30.0
    basis = gaussian_basis(wl, centers, widths)
    
    # Generate synthetic spectra
    n_objs = 10
    spectra_true, weights_true = generate_synthetic_spectra(n_objs, basis, rng=42)
    
    # Define filters (B, V, R)
    filters = [
        ("B", create_tophat_filter(wl, 400, 500)),
        ("V", create_tophat_filter(wl, 500, 600)),
        ("R", create_tophat_filter(wl, 600, 700)),
    ]
    
    # Compute photometric fluxes
    phot = compute_photometry(spectra_true, wl, filters)
    
    # Reconstruct coefficients from photometry
    coeffs_recon = reconstruct_weights(phot, basis, wl, filters, alpha=0.1)
    
    # Reconstruct spectra
    spectra_recon = reconstruct_spectra(coeffs_recon, basis)
    
    # Compare first object
    obj_idx = 0
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(wl, spectra_true[obj_idx], label="True")
    plt.plot(wl, spectra_recon[obj_idx], '--', label="Reconstructed")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Synthetic Spectrum Reconstruction")
    plt.show()


if __name__ == "__main__":
    main()