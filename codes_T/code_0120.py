import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# ---------- Spectral model ----------
def wavelength_grid(start=400.0, stop=800.0, npoints=1000):
    """Generate a wavelength array in nanometers."""
    return np.linspace(start, stop, npoints)

def generate_basis(n_basis, wl):
    """
    Generate an orthogonal basis for spectra.
    Each column is a basis function defined over the wavelength array.
    """
    rng = np.random.default_rng()
    random_mat = rng.standard_normal((wl.size, n_basis))
    # Orthogonalize columns using QR decomposition
    q, _ = np.linalg.qr(random_mat)
    return q[:, :n_basis]

def spectral_model(coeffs, basis):
    """
    Construct a spectrum from linear combination of basis functions.
    """
    return basis @ coeffs

# ---------- Filters ----------
def gaussian_filter(center, width, wl):
    """Single Gaussian filter transmission."""
    return norm.pdf(wl, loc=center, scale=width)

def generate_filters(n_filters, wl):
    """
    Generate a set of Gaussian filters with random centers and widths.
    """
    rng = np.random.default_rng()
    centers = rng.uniform(wl.min(), wl.max(), size=n_filters)
    widths = rng.uniform(20.0, 60.0, size=n_filters)
    filters = np.array([gaussian_filter(c, w, wl) for c, w in zip(centers, widths)])
    return filters

# ---------- Photometry ----------
def compute_photometry(spectrum, filters, wl):
    """
    Integrate spectrum over each filter to obtain photometric fluxes.
    Normalise by the integral of the filter transmission.
    """
    fluxes = []
    for filt in filters:
        num = np.trapz(spectrum * filt, wl)
        den = np.trapz(filt, wl)
        fluxes.append(num / den)
    return np.array(fluxes)

# ---------- Reconstruction ----------
def construct_design_matrix(filters, basis, wl):
    """
    Compute the matrix that maps basis coefficients to photometric fluxes.
    Each row corresponds to a filter, each column to a basis function.
    """
    design = np.empty((filters.shape[0], basis.shape[1]))
    for i, filt in enumerate(filters):
        # Integral of basis_j * filt over wavelength
        design[i, :] = np.trapz(basis * filt, wl, axis=0) / np.trapz(filt, wl)
    return design

def reconstruct_spectrum(photometry, filters, basis, wl, method='ols'):
    """
    Reconstruct spectrum coefficients from photometric fluxes.
    Supported methods: 'ols' (ordinary least squares), 'ridge'.
    """
    A = construct_design_matrix(filters, basis, wl)
    if method == 'ols':
        lr = LinearRegression(fit_intercept=False)
        lr.fit(A, photometry)
        coeffs = lr.coef_
    else:
        raise ValueError("Unsupported reconstruction method.")
    reconstructed = basis @ coeffs
    return coeffs, reconstructed

# ---------- Example usage ----------
if __name__ == "__main__":
    # Define wavelength array
    wl = wavelength_grid()

    # Generate orthogonal basis
    n_basis = 5
    basis = generate_basis(n_basis, wl)

    # Generate synthetic spectrum
    rng = np.random.default_rng()
    true_coeffs = rng.normal(size=n_basis)
    true_spectrum = spectral_model(true_coeffs, basis)

    # Generate filter set
    n_filters = 4
    filters = generate_filters(n_filters, wl)

    # Compute synthetic photometry
    photometry = compute_photometry(true_spectrum, filters, wl)

    # Reconstruct spectrum from photometry
    rec_coeffs, rec_spectrum = reconstruct_spectrum(photometry, filters, basis, wl)

    # Evaluate reconstruction error
    error = np.linalg.norm(rec_spectrum - true_spectrum) / np.linalg.norm(true_spectrum)
    print(f"Reconstruction relative L2 error: {error:.4f}")