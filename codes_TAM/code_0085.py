#!/usr/bin/env python3
import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Define a simple spectral model: a linear combination of Gaussians
# ----------------------------------------------------------------------
def gaussian_basis(wavelengths, n_bases=5, rng=None):
    """
    Construct n_bases Gaussian basis functions over the wavelength grid.

    Parameters
    ----------
    wavelengths : ndarray
        1D array of wavelength values.
    n_bases : int
        Number of Gaussian basis functions.
    rng : np.random.Generator or None
        Random number generator for reproducible centers and widths.

    Returns
    -------
    basis : ndarray
        Shape (n_bases, len(wavelengths))
    """
    rng = rng or np.random.default_rng()
    centers = rng.uniform(wavelengths.min(), wavelengths.max(), size=n_bases)
    widths = rng.uniform((wavelengths.max()-wavelengths.min())/20,
                         (wavelengths.max()-wavelengths.min())/10,
                         size=n_bases)
    # Evaluate Gaussians
    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :])**2)
    return basis

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(basis, n_spectra=10, rng=None):
    """
    Generate synthetic spectra as random linear combinations of basis functions.

    Parameters
    ----------
    basis : ndarray
        Shape (n_bases, n_wavelengths)
    n_spectra : int
        Number of spectra to generate.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    spectra : ndarray
        Shape (n_spectra, n_wavelengths)
    coeffs : ndarray
        Shape (n_spectra, n_bases)  (true coefficients)
    """
    rng = rng or np.random.default_rng()
    coeffs = rng.normal(size=(n_spectra, basis.shape[0]))
    spectra = coeffs @ basis  # linear combination
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Generate photometric filter responses
# ----------------------------------------------------------------------
def create_tophat_filter(wavelengths, center, width):
    """
    Create a simple top-hat filter transmission curve.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength grid.
    center : float
        Center wavelength of the filter.
    width : float
        Full width at half maximum (top-hat width).

    Returns
    -------
    response : ndarray
        Transmission curve (0 or 1).
    """
    return np.where(np.abs(wavelengths - center) <= width / 2, 1.0, 0.0)

def generate_filters(wavelengths, filter_specs):
    """
    Build an array of filter transmission curves.

    Parameters
    ----------
    wavelengths : ndarray
        Wavelength grid.
    filter_specs : list of tuples
        Each tuple is (center, width).

    Returns
    -------
    filters : ndarray
        Shape (n_filters, len(wavelengths))
    """
    return np.array([create_tophat_filter(wavelengths, c, w)
                     for c, w in filter_specs])

# ----------------------------------------------------------------------
# 4. Compute photometric fluxes from spectra
# ----------------------------------------------------------------------
def compute_photometry(spectra, filters, wavelengths):
    """
    Integrate spectra over each filter bandpass.

    Parameters
    ----------
    spectra : ndarray
        Shape (n_spectra, n_wavelengths)
    filters : ndarray
        Shape (n_filters, n_wavelengths)
    wavelengths : ndarray
        Wavelength grid.

    Returns
    -------
    fluxes : ndarray
        Shape (n_spectra, n_filters)
    """
    # Integrate product of spectrum and filter response
    fluxes = np.array([simps(s * f, wavelengths) for s in spectra for f in filters])
    return fluxes.reshape(spectra.shape[0], filters.shape[0])

# ----------------------------------------------------------------------
# 5. Reconstruct spectra from photometry
# ----------------------------------------------------------------------
def build_design_matrix(filters, basis, wavelengths):
    """
    Construct the matrix that maps coefficients to photometric fluxes.

    Parameters
    ----------
    filters : ndarray
        Shape (n_filters, n_wavelengths)
    basis : ndarray
        Shape (n_bases, n_wavelengths)
    wavelengths : ndarray
        Wavelength grid.

    Returns
    -------
    design : ndarray
        Shape (n_filters, n_bases)
    """
    # For each filter, integrate each basis function over the filter
    design = np.array([
        [simps(b * f, wavelengths) for b in basis]
        for f in filters
    ])
    return design

def reconstruct_coefficients(fluxes, design_matrix):
    """
    Solve linear least-squares to recover coefficients.

    Parameters
    ----------
    fluxes : ndarray
        Observed photometric fluxes, shape (n_spectra, n_filters).
    design_matrix : ndarray
        Mapping from coefficients to photometry, shape (n_filters, n_bases).

    Returns
    -------
    coeffs_est : ndarray
        Estimated coefficients, shape (n_spectra, n_bases).
    """
    lr = LinearRegression(fit_intercept=False)
    lr.fit(design_matrix.T, fluxes.T)
    return lr.coef_.T

def reconstruct_spectra(coeffs_est, basis):
    """
    Reconstruct spectra from estimated coefficients.

    Parameters
    ----------
    coeffs_est : ndarray
        Estimated coefficients, shape (n_spectra, n_bases).
    basis : ndarray
        Basis functions, shape (n_bases, n_wavelengths).

    Returns
    -------
    spectra_rec : ndarray
        Reconstructed spectra, shape (n_spectra, n_wavelengths).
    """
    return coeffs_est @ basis

# ----------------------------------------------------------------------
# 6. Example usage
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)

    # Wavelength grid (nm)
    wavelengths = np.linspace(300, 800, 200)

    # Spectral basis
    basis = gaussian_basis(wavelengths, n_bases=5, rng=rng)

    # Synthetic spectra
    n_spectra = 8
    spectra_true, coeffs_true = generate_synthetic_spectra(basis, n_spectra, rng=rng)

    # Filters: three top-hat filters
    filter_specs = [(350, 50), (550, 70), (700, 60)]
    filters = generate_filters(wavelengths, filter_specs)

    # Photometric fluxes
    fluxes_obs = compute_photometry(spectra_true, filters, wavelengths)

    # Design matrix
    design = build_design_matrix(filters, basis, wavelengths)

    # Reconstruct coefficients
    coeffs_rec = reconstruct_coefficients(fluxes_obs, design)

    # Reconstruct spectra
    spectra_rec = reconstruct_spectra(coeffs_rec, basis)

    # Compare true vs reconstructed spectra for first spectrum
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectra_true[0], label='True spectrum')
    plt.plot(wavelengths, spectra_rec[0], '--', label='Reconstructed')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('Spectral Reconstruction Example')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()