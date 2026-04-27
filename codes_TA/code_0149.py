import numpy as np
from scipy import interpolate
from sklearn.linear_model import Ridge

# --------------------------------------------------------------------------- #
# 1. Define spectral model – sum of Gaussian lines
# --------------------------------------------------------------------------- #
def gaussian_spectrum(wavelength, amps, centers, widths):
    """Return a spectrum as a sum of Gaussians."""
    spec = np.zeros_like(wavelength)
    for a, c, w in zip(amps, centers, widths):
        spec += a * np.exp(-0.5 * ((wavelength - c) / w) ** 2)
    return spec

# --------------------------------------------------------------------------- #
# 2. Generate synthetic spectra
# --------------------------------------------------------------------------- #
def generate_synthetic_spectra(n_samples, wavelength_grid, n_lines=3, rng=None):
    rng = np.random.default_rng(rng)
    spectra = []
    for _ in range(n_samples):
        amps   = rng.uniform(0.5, 1.5, size=n_lines)
        centers = rng.uniform(4000, 8000, size=n_lines)  # Angstroms
        widths  = rng.uniform(50, 200, size=n_lines)
        spectra.append(gaussian_spectrum(wavelength_grid, amps, centers, widths))
    return np.array(spectra)  # shape (n_samples, n_pixels)

# --------------------------------------------------------------------------- #
# 3. Create filter transmission curves (Gaussian filters)
# --------------------------------------------------------------------------- #
def create_filters(n_filters, wavelength_grid, rng=None):
    rng = np.random.default_rng(rng)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(4500, 7500)
        width  = rng.uniform(100, 300)
        trans = np.exp(-0.5 * ((wavelength_grid - center) / width) ** 2)
        trans /= trans.sum()            # normalise to unit integral
        filters.append(trans)
    return np.array(filters)  # shape (n_filters, n_pixels)

# --------------------------------------------------------------------------- #
# 4. Compute photometric fluxes from spectra
# --------------------------------------------------------------------------- #
def compute_photometry(spectra, filters):
    """
    spectra : (n_samples, n_pixels)
    filters : (n_filters, n_pixels)
    returns photometry: (n_samples, n_filters)
    """
    return spectra @ filters.T   # matrix multiplication

# --------------------------------------------------------------------------- #
# 5. Reconstruct spectra from photometry
# --------------------------------------------------------------------------- #
def reconstruct_spectra(photometry, filters, n_basis=5, alpha=1e-2):
    """
    Reconstruct each spectrum as a linear combination of basis Gaussians.
    Basis functions are simple Gaussians centred at fixed wavelengths.
    """
    n_samples, n_filters = photometry.shape
    n_pixels = filters.shape[1]

    # Define basis functions on wavelength grid
    basis_centers = np.linspace(4000, 8000, n_basis)
    basis_width   = 150.0
    basis_funcs = [np.exp(-0.5 * ((np.arange(n_pixels) - center) / basis_width) ** 2)
                   for center in basis_centers]
    basis_matrix = np.stack(basis_funcs, axis=1)  # shape (n_pixels, n_basis)

    # Build design matrix: photometry = (basis_matrix.T @ filters.T) @ coeffs
    # So we first compute the mapping from basis coefficients to photometry
    # M = filters @ basis_matrix  => (n_filters, n_basis)
    M = filters @ basis_matrix  # shape (n_filters, n_basis)

    # Fit ridge regression: coeffs = (M^T M + alpha I)^-1 M^T photometry
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(M, photometry)
    coeffs = reg.coef_.T  # shape (n_basis, n_samples)

    # Reconstruct spectra
    reconstructed = basis_matrix @ coeffs  # shape (n_pixels, n_samples)
    return reconstructed.T  # shape (n_samples, n_pixels)

# --------------------------------------------------------------------------- #
# 6. Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Wavelength grid (Angstroms)
    wl_grid = np.linspace(3500, 9500, 1200)

    # Generate data
    spectra = generate_synthetic_spectra(n_samples=20, wavelength_grid=wl_grid, rng=42)
    filters = create_filters(n_filters=5, wavelength_grid=wl_grid, rng=24)

    # Photometry
    phot = compute_photometry(spectra, filters)

    # Reconstruction
    recon = reconstruct_spectra(phot, filters, n_basis=7, alpha=1e-1)

    # Simple evaluation: mean squared error between true and reconstructed spectra
    mse = np.mean((spectra - recon) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")

    # Visual inspection (optional, uncomment if running locally)
    # import matplotlib.pyplot as plt
    # idx = 0
    # plt.figure(figsize=(8,4))
    # plt.plot(wl_grid, spectra[idx], label='True')
    # plt.plot(wl_grid, recon[idx], '--', label='Reconstructed')
    # plt.xlabel('Wavelength (Å)')
    # plt.ylabel('Flux (arb. units)')
    # plt.legend()
    # plt.show()