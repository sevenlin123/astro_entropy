import numpy as np
from scipy.signal import gaussian
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Spectral model definition
# -----------------------------
def spectral_basis(wave, n_basis=4):
    """
    Construct a basis matrix for spectra.
    Basis consists of:
        - a constant term
        - n_basis-1 Gaussian peaks with fixed widths
    Parameters
    ----------
    wave : ndarray, shape (nw,)
        Wavelength grid.
    n_basis : int
        Number of basis functions.
    Returns
    -------
    B : ndarray, shape (nw, n_basis)
        Basis matrix.
    """
    nw = len(wave)
    B = np.zeros((nw, n_basis))
    # constant term
    B[:, 0] = 1.0
    # Gaussian peaks
    centers = np.linspace(wave[0], wave[-1], n_basis-1)
    sigma = 200.0  # angstrom
    for i, cen in enumerate(centers, start=1):
        B[:, i] = np.exp(-0.5 * ((wave - cen)/sigma)**2)
    return B


# -----------------------------
# 2. Generate synthetic spectra
# -----------------------------
def generate_synthetic_spectra(n_spectra, wave, rng=None):
    """
    Generate synthetic spectra as random linear combinations of the basis.
    Parameters
    ----------
    n_spectra : int
        Number of spectra to generate.
    wave : ndarray
        Wavelength grid.
    rng : np.random.Generator, optional
        Random number generator.
    Returns
    -------
    spectra : ndarray, shape (n_spectra, nw)
    coeffs : ndarray, shape (n_spectra, n_basis)
    """
    if rng is None:
        rng = np.random.default_rng()
    B = spectral_basis(wave)
    n_basis = B.shape[1]
    coeffs = rng.uniform(low=0.5, high=1.5, size=(n_spectra, n_basis))
    spectra = coeffs @ B.T
    return spectra, coeffs


# -----------------------------
# 3. Define photometric filters
# -----------------------------
def define_filters(n_filters, wave, rng=None):
    """
    Create simple top‑hat filters.
    Parameters
    ----------
    n_filters : int
        Number of filters.
    wave : ndarray
        Wavelength grid.
    rng : np.random.Generator, optional
        Random number generator.
    Returns
    -------
    filters : ndarray, shape (n_filters, nw)
    """
    if rng is None:
        rng = np.random.default_rng()
    nw = len(wave)
    filters = np.zeros((n_filters, nw))
    # Randomly choose central wavelengths and widths
    centroids = rng.uniform(wave[0]+300, wave[-1]-300, size=n_filters)
    widths = rng.uniform(200, 600, size=n_filters)
    for i, (cen, wid) in enumerate(zip(centroids, widths)):
        mask = np.abs(wave - cen) < wid/2.0
        filters[i, mask] = 1.0
    # Normalise filters
    filters /= np.sum(filters, axis=1, keepdims=True)
    return filters


# -----------------------------
# 4. Compute photometry
# -----------------------------
def compute_photometry(spectra, filters):
    """
    Integrate spectra through filters to obtain photometric fluxes.
    Parameters
    ----------
    spectra : ndarray, shape (n_spectra, nw)
    filters : ndarray, shape (n_filters, nw)
    Returns
    -------
    photometry : ndarray, shape (n_spectra, n_filters)
    """
    # Simple trapezoidal integration over wavelength
    # Assume uniform spacing for simplicity
    return spectra @ filters.T


# -----------------------------
# 5. Reconstruct spectra
# -----------------------------
def reconstruct_spectra(photometry, wave, filters):
    """
    Reconstruct spectra from photometry by solving least squares
    for the basis coefficients.
    Parameters
    ----------
    photometry : ndarray, shape (n_spectra, n_filters)
    wave : ndarray
    filters : ndarray, shape (n_filters, nw)
    Returns
    -------
    reconstructed_spectra : ndarray, shape (n_spectra, nw)
    estimated_coeffs : ndarray, shape (n_spectra, n_basis)
    """
    B = spectral_basis(wave)
    # Build mapping matrix G = F * B^T
    G = filters @ B.T  # shape (n_filters, n_basis)
    n_spectra = photometry.shape[0]
    est_coeffs = np.zeros((n_spectra, G.shape[1]))
    for i in range(n_spectra):
        # Solve G * c = p_i
        est_coeffs[i] = np.linalg.lstsq(G, photometry[i], rcond=None)[0]
    recon_spectra = est_coeffs @ B.T
    return recon_spectra, est_coeffs


# -----------------------------
# 6. Demo
# -----------------------------
def main():
    rng = np.random.default_rng(seed=42)
    # Wavelength grid (4000–8000 Å)
    wave = np.linspace(4000, 8000, 1000)
    # Generate synthetic data
    spectra, true_coeffs = generate_synthetic_spectra(10, wave, rng=rng)
    filters = define_filters(5, wave, rng=rng)
    photometry = compute_photometry(spectra, filters)
    # Reconstruction
    recon_spectra, est_coeffs = reconstruct_spectra(photometry, wave, filters)
    # Print comparison
    for i in range(3):
        print(f"Sample {i+1}")
        print("  True coeffs:", true_coeffs[i])
        print("  Estimated coeffs:", est_coeffs[i])
        err = np.linalg.norm(recon_spectra[i] - spectra[i]) / np.linalg.norm(spectra[i])
        print(f"  Relative reconstruction error: {err:.3f}\n")

if __name__ == "__main__":
    main()