import numpy as np
from scipy.optimize import least_squares

# ----------------------------------------------------------------------
# 1. Define a simple spectral model
# ----------------------------------------------------------------------
def gaussian(x, mu, sigma):
    """One-dimensional Gaussian."""
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def create_basis(n_basis=10, wlen=np.linspace(400, 800, 400)):
    """
    Build a set of Gaussian basis functions evenly spaced over the wavelength grid.
    Returns:
        basis: array of shape (n_basis, len(wlen))
    """
    mus = np.linspace(wlen[0], wlen[-1], n_basis)
    sigma = (wlen[-1] - wlen[0]) / (2 * n_basis)  # width
    basis = np.array([gaussian(wlen, mu, sigma) for mu in mus])
    return basis

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_spectrum(basis, coef=None):
    """
    Create a synthetic spectrum as a linear combination of basis functions.
    If coef is None, random coefficients are chosen.
    """
    if coef is None:
        coef = np.random.randn(basis.shape[0])
    spectrum = basis.T @ coef
    return spectrum, coef

# ----------------------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# ----------------------------------------------------------------------
def create_filters(n_filters=5, wlen=np.linspace(400, 800, 400)):
    """
    Create simple top-hat filter transmission curves.
    Returns an array of shape (n_filters, len(wlen)).
    """
    filters = []
    widths = (wlen[-1] - wlen[0]) / (3 * n_filters)
    centers = np.linspace(wlen[0] + widths, wlen[-1] - widths, n_filters)
    for c in centers:
        filt = np.where((wlen >= c - widths/2) & (wlen <= c + widths/2), 1.0, 0.0)
        filters.append(filt)
    return np.array(filters)

def compute_photometry(spectrum, filters, wlen):
    """
    Integrate the spectrum over each filter to produce photometric fluxes.
    """
    fluxes = np.trapz(spectrum * filters, x=wlen, axis=1)
    return fluxes

# ----------------------------------------------------------------------
# 4. Reconstruct a synthetic spectrum from photometric data
# ----------------------------------------------------------------------
def build_design_matrix(basis, filters, wlen):
    """
    Build the matrix that maps coefficients to photometric fluxes.
    Each row corresponds to a filter, each column to a basis function.
    """
    A = np.trapz(basis * filters[:, :, None], x=wlen, axis=2)
    return A.T  # shape (n_filters, n_basis)

def reconstruct_spectrum(A, fluxes, wlen, n_basis):
    """
    Solve for coefficients via linear least squares given design matrix A and photometric fluxes.
    Then rebuild the full spectrum on the wavelength grid.
    """
    coef, *_ = np.linalg.lstsq(A, fluxes, rcond=None)
    # Reconstruct spectrum from coefficients
    basis = np.array([gaussian(wlen, mu, (wlen[-1]-wlen[0])/(2*n_basis))
                      for mu in np.linspace(wlen[0], wlen[-1], n_basis)])
    reconstructed = basis.T @ coef
    return reconstructed, coef

# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Wavelength grid
    wlen = np.linspace(400, 800, 400)

    # Basis
    n_basis = 10
    basis = create_basis(n_basis=n_basis, wlen=wlen)

    # True spectrum
    true_spectrum, true_coef = generate_spectrum(basis)

    # Filters
    n_filters = 5
    filters = create_filters(n_filters=n_filters, wlen=wlen)

    # Photometry
    fluxes = compute_photometry(true_spectrum, filters, wlen)

    # Design matrix
    A = build_design_matrix(basis, filters, wlen)

    # Reconstruct spectrum
    recon_spectrum, recon_coef = reconstruct_spectrum(A, fluxes, wlen, n_basis)

    # Simple sanity check
    print("True coefficients  :", true_coef)
    print("Recovered coeffs   :", recon_coef)
    print("True vs Recovered:")
    print(np.abs(true_coef - recon_coef))