import numpy as np
from numpy.linalg import lstsq
from scipy.interpolate import interp1d

def create_wavelength_grid(n_wave=1000, wl_min=300.0, wl_max=2500.0):
    """Create a regular wavelength grid in nanometers."""
    return np.linspace(wl_min, wl_max, n_wave)

def create_basis_spectra(wavelength, n_basis=5, seed=42):
    """Generate smooth basis spectra using Gaussian mixtures."""
    rng = np.random.default_rng(seed)
    bases = []
    for i in range(n_basis):
        # Random centers and widths
        centers = rng.uniform(wavelength.min(), wavelength.max(), size=3)
        widths = rng.uniform(30, 200, size=3)
        amps = rng.uniform(0.5, 1.5, size=3)
        spec = sum(a * np.exp(-0.5 * ((w - c) / s)**2)
                   for a, c, s in zip(amps, centers, widths))
        # Normalize to unit integral
        spec /= np.trapz(spec, wavelength)
        bases.append(spec)
    return np.array(bases)  # shape (n_basis, n_wave)

def create_filters(wavelength, n_filters=4, seed=123):
    """Create simple boxcar filter transmission curves."""
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wavelength.min() + 50, wavelength.max() - 50)
        width = rng.uniform(50, 150)
        filt = np.where(np.abs(wavelength - center) <= width/2, 1.0, 0.0)
        filters.append(filt)
    return np.array(filters)  # shape (n_filters, n_wave)

def generate_random_coefficients(n_samples, n_basis, seed=7):
    """Generate non‑negative random coefficients for each synthetic spectrum."""
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(0.0, 5.0, size=(n_samples, n_basis))
    return coeffs

def compute_filter_matrix(basis, filters, wavelength):
    """
    Compute the matrix relating coefficients to photometric fluxes:
    F_i = sum_k coeff_k * int(basis_k * filter_i)
    Returns a matrix of shape (n_filters, n_basis).
    """
    mat = np.zeros((filters.shape[0], basis.shape[0]))
    for i, filt in enumerate(filters):
        for k, bas in enumerate(basis):
            mat[i, k] = np.trapz(bas * filt, wavelength)
    return mat

def generate_photometry(coeffs, filter_matrix):
    """Compute synthetic photometric measurements."""
    return coeffs @ filter_matrix.T  # shape (n_samples, n_filters)

def reconstruct_coefficients(photometry, filter_matrix, basis, wavelength):
    """
    Reconstruct coefficients by least‑squares inversion.
    Returns reconstructed spectra and coefficients.
    """
    # Solve for coefficients: filter_matrix^T * coeffs^T = photometry^T
    coeffs_rec, *_ = lstsq(filter_matrix.T, photometry.T, rcond=None)
    coeffs_rec = coeffs_rec.T  # shape (n_samples, n_basis)
    spectra_rec = coeffs_rec @ basis  # shape (n_samples, n_wave)
    return coeffs_rec, spectra_rec

def main():
    # 1. Define wavelength grid
    wl = create_wavelength_grid()

    # 2. Define spectral model (basis spectra)
    basis = create_basis_spectra(wl, n_basis=5)

    # 3. Generate synthetic spectra
    n_samples = 10
    coeffs_true = generate_random_coefficients(n_samples, basis.shape[0])
    spectra_true = coeffs_true @ basis  # shape (n_samples, n_wave)

    # 4. Generate photometric data from synthetic spectra
    filters = create_filters(wl, n_filters=4)
    filter_mat = compute_filter_matrix(basis, filters, wl)
    photometry = generate_photometry(coeffs_true, filter_mat)

    # 5. Reconstruct spectra from photometry
    coeffs_est, spectra_est = reconstruct_coefficients(
        photometry, filter_mat, basis, wl
    )

    # Print reconstruction error for demonstration
    rms_error = np.sqrt(((spectra_true - spectra_est)**2).mean(axis=1))
    print("RMS reconstruction error per sample:", rms_error)

if __name__ == "__main__":
    main()