import numpy as np
from scipy.integrate import trapz
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model: 5 Gaussian basis functions
# ----------------------------------------------------------------------
def gaussian_basis(wl, centers, widths):
    """
    Construct a set of Gaussian basis functions.
    Parameters
    ----------
    wl : ndarray, shape (N,)
        Wavelength grid.
    centers : list or ndarray, shape (M,)
        Centers of the Gaussian basis functions.
    widths : list or ndarray, shape (M,)
        Standard deviations of the Gaussians.
    Returns
    -------
    basis : ndarray, shape (N, M)
        Basis function matrix.
    """
    wl = wl[:, None]          # (N, 1)
    centers = np.asarray(centers)[None, :]  # (1, M)
    widths  = np.asarray(widths)[None, :]   # (1, M)
    gauss = np.exp(-0.5 * ((wl - centers) / widths)**2)
    return gauss


def synthetic_spectrum(wl, coeffs, centers, widths):
    """
    Generate a synthetic spectrum as a linear combination
    of Gaussian basis functions.
    """
    basis = gaussian_basis(wl, centers, widths)
    return basis @ coeffs


# ----------------------------------------------------------------------
# Photometric system: simple Gaussian passband filters
# ----------------------------------------------------------------------
def gaussian_filter(wl, center, width):
    """
    Create a Gaussian filter transmission curve.
    """
    return np.exp(-0.5 * ((wl - center) / width)**2)


def photometry_from_spectrum(spectrum, wl, filters):
    """
    Compute synthetic photometry by integrating the product
    of spectrum and filter transmissions.
    Parameters
    ----------
    spectrum : ndarray, shape (N,)
    wl : ndarray, shape (N,)
    filters : list of tuples [(center, width), ...]
    Returns
    -------
    fluxes : ndarray, shape (len(filters),)
    """
    fluxes = []
    for center, width in filters:
        trans = gaussian_filter(wl, center, width)
        flux = trapz(spectrum * trans, wl)
        fluxes.append(flux)
    return np.array(fluxes)


# ----------------------------------------------------------------------
# Reconstruction from photometry
# ----------------------------------------------------------------------
def reconstruct_spectrum(photometry, wl, filters,
                         centers, widths, reg_alpha=1e-3):
    """
    Reconstruct a spectrum by fitting the basis coefficients
    that reproduce the given photometry.
    Uses Ridge regression (least‑squares + L2 regularisation).
    Parameters
    ----------
    photometry : ndarray, shape (F,)
    wl : ndarray, shape (N,)
    filters : list of tuples [(center, width), ...]
    centers, widths : basis hyperparameters
    reg_alpha : float
        Regularisation strength.
    Returns
    -------
    coeffs : ndarray, shape (M,)
        Fitted basis coefficients.
    reconstructed_spectrum : ndarray, shape (N,)
    """
    # Build matrix A where A_ij = ∫ basis_j * filter_i dλ
    basis = gaussian_basis(wl, centers, widths)            # (N, M)
    A = []
    for center, width in filters:
        trans = gaussian_filter(wl, center, width)         # (N,)
        integrand = basis * trans[:, None]                 # (N, M)
        integral = trapz(integrand, wl, axis=0)            # (M,)
        A.append(integral)
    A = np.vstack(A)                                       # (F, M)

    # Fit coefficients using Ridge regression
    clf = Ridge(alpha=reg_alpha, fit_intercept=False)
    clf.fit(A, photometry)
    coeffs = clf.coef_
    reconstructed = basis @ coeffs
    return coeffs, reconstructed


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Wavelength grid (400-800 nm)
    wl = np.linspace(400, 800, 2000)           # (N,)

    # Basis hyperparameters
    centers = [450, 500, 550, 600, 650]        # (M,)
    widths  = [20, 20, 20, 20, 20]             # (M,)

    # Random true coefficients
    true_coeffs = np.random.randn(len(centers))

    # Generate synthetic spectrum
    true_spectrum = synthetic_spectrum(wl, true_coeffs, centers, widths)

    # Define 5 photometric filters (center, width)
    filters = [(440, 30), (530, 30), (610, 30), (690, 30), (770, 30)]

    # Compute synthetic photometry
    phot = photometry_from_spectrum(true_spectrum, wl, filters)

    # Reconstruct spectrum from photometry
    coeffs_rec, rec_spectrum = reconstruct_spectrum(
        phot, wl, filters, centers, widths, reg_alpha=1e-2
    )

    # Print results
    print("True coefficients :", true_coeffs)
    print("Reconstructed coeffs:", coeffs_rec)
    print("\nFirst 10 values of true spectrum:")
    print(true_spectrum[:10])
    print("\nFirst 10 values of reconstructed spectrum:")
    print(rec_spectrum[:10])