import numpy as np
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Basic utilities -------------------------------------------------------
def wavelength_grid(start_nm, stop_nm, num):
    """Create an evenly spaced wavelength array."""
    return np.linspace(start_nm, stop_nm, num)

def gaussian(x, mu, sigma):
    """Simple Gaussian function."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# ----------------------------------------------------------------------
# Spectral model --------------------------------------------------------
def create_basis_spectra(wl, n_basis=5):
    """
    Construct a set of basis spectra (Gaussian bumps).
    Returns an array of shape (n_basis, len(wl)).
    """
    rng = np.random.default_rng(seed=42)
    mus = np.linspace(wl[0], wl[-1], n_basis)
    sigmas = np.full(n_basis, (wl[-1] - wl[0]) / (2 * n_basis))
    basis = np.array([gaussian(wl, mu, sigma) for mu, sigma in zip(mus, sigmas)])
    # normalize each basis spectrum
    basis /= basis.sum(axis=1, keepdims=True)
    return basis

def create_filters(wl, centers, widths):
    """
    Generate a set of filter transmission curves.
    Returns a list of arrays, each same shape as wl.
    """
    filters = []
    for c, w in zip(centers, widths):
        filt = gaussian(wl, c, w / 2.355)  # convert FWHM to sigma
        filt /= filt.max()                 # normalize
        filters.append(filt)
    return filters

# ----------------------------------------------------------------------
# Forward model ---------------------------------------------------------
def compute_filter_matrix(basis, filters, wl):
    """
    Compute matrix F such that photometry = F @ coefficients.
    Basis shape: (n_basis, n_wl)
    Filters is a list of arrays, each shape (n_wl,)
    Returns F of shape (n_filters, n_basis)
    """
    n_filters = len(filters)
    n_basis = basis.shape[0]
    F = np.zeros((n_filters, n_basis))
    dl = wl[1] - wl[0]   # wavelength step
    for i, filt in enumerate(filters):
        for j in range(n_basis):
            # integral of basis[j] * filt over wavelength
            F[i, j] = np.trapz(basis[j] * filt, wl) * dl
    return F

def generate_synthetic_spectra(n_samples, basis):
    """
    Generate random linear combinations of basis spectra.
    Returns spectra (n_samples, n_wl) and the coefficient matrix.
    """
    rng = np.random.default_rng(seed=123)
    n_basis = basis.shape[0]
    coeffs = rng.normal(size=(n_samples, n_basis))
    spectra = coeffs @ basis
    return spectra, coeffs

def compute_photometry(spectra, filters, wl):
    """
    Integrate spectra through filters to obtain photometric fluxes.
    """
    n_samples = spectra.shape[0]
    n_filters = len(filters)
    phot = np.zeros((n_samples, n_filters))
    dl = wl[1] - wl[0]
    for i, filt in enumerate(filters):
        # elementwise product and integral
        phot[:, i] = np.trapz(spectra * filt, wl, axis=1) * dl
    return phot

# ----------------------------------------------------------------------
# Reconstruction framework ---------------------------------------------
def reconstruct_coefficients(photometry, F, alpha=1e-3):
    """
    Recover spectral coefficients from photometric data.
    Uses ridge regression to regularize the inverse problem.
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(F.T, photometry.T)
    return ridge.coef_.T   # shape (n_samples, n_basis)

def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from coefficients and basis."""
    return coeffs @ basis

# ----------------------------------------------------------------------
# Demo ---------------------------------------------------------------
def main():
    # Wavelength grid
    wl = wavelength_grid(400, 1000, 500)  # nm

    # Basis spectra
    basis = create_basis_spectra(wl, n_basis=5)

    # Photometric filters
    filter_centers = [450, 550, 650, 750]  # nm
    filter_widths = [50, 50, 50, 50]       # nm (FWHM)
    filters = create_filters(wl, filter_centers, filter_widths)

    # Filter matrix
    F = compute_filter_matrix(basis, filters, wl)

    # Synthetic data
    n_objects = 20
    spectra, true_coeffs = generate_synthetic_spectra(n_objects, basis)

    # Photometry
    phot = compute_photometry(spectra, filters, wl)

    # Reconstruction
    rec_coeffs = reconstruct_coefficients(phot, F)
    rec_spectra = reconstruct_spectra(rec_coeffs, basis)

    # Evaluate reconstruction error
    rmse = np.sqrt(((spectra - rec_spectra) ** 2).mean(axis=1))
    print("RMSE per object:", rmse)
    print("Average RMSE:", rmse.mean())

if __name__ == "__main__":
    main()