import numpy as np
from scipy import special
from sklearn.linear_model import Ridge

# ----- Spectral model -----
def gaussian_basis(wavelength, centers, widths):
    """Generate Gaussian basis functions."""
    return np.exp(-0.5 * ((wavelength[:, None] - centers[None, :]) / widths[None, :])**2)

# ----- Synthetic spectra -----
def synthesize_spectrum(coeffs, basis):
    """Linear combination of basis functions."""
    return basis.T @ coeffs

# ----- Filter curves -----
def gaussian_filter(wavelength, center, width, amplitude=1.0):
    """Single Gaussian filter transmission."""
    return amplitude * np.exp(-0.5 * ((wavelength - center) / width)**2)

def generate_random_filters(n_filters, wavelength, rng=None):
    """Generate a list of random Gaussian filters."""
    if rng is None:
        rng = np.random.default_rng()
    centers = rng.uniform(wavelength.min(), wavelength.max(), size=n_filters)
    widths = rng.uniform((wavelength.max()-wavelength.min())/20,
                         (wavelength.max()-wavelength.min())/10,
                         size=n_filters)
    filters = [gaussian_filter(wavelength, c, w) for c, w in zip(centers, widths)]
    return np.array(filters)  # shape (n_filters, n_wavelength)

# ----- Photometry -----
def compute_photometry(spectrum, filters, wavelength):
    """Compute integrated fluxes through each filter."""
    return np.trapz(spectrum[:, None] * filters, x=wavelength, axis=0)

# ----- Reconstruction -----
def construct_weight_matrix(filters, basis, wavelength):
    """W_ij = integral(basis_j * filter_i)"""
    return np.trapz(basis[:, None, :] * filters[:, None, :], x=wavelength, axis=2)

def reconstruct_coefficients(photometry, W, alpha=1.0):
    """Solve W c = photometry for coefficients via ridge regression."""
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(W, photometry)
    return reg.coef_

# ----- Demo -----
def main():
    rng = np.random.default_rng(42)

    # Wavelength grid
    n_wave = 1000
    wave = np.linspace(400, 800, n_wave)  # nm

    # Basis functions
    n_basis = 30
    centers = np.linspace(wave.min(), wave.max(), n_basis)
    widths = np.full(n_basis, (wave.max()-wave.min())/25)
    basis = gaussian_basis(wave, centers, widths)

    # True coefficients
    true_coeffs = rng.normal(size=n_basis)

    # Generate true spectrum
    true_spectrum = synthesize_spectrum(true_coeffs, basis)

    # Generate filters
    n_filters = 7
    filters = generate_random_filters(n_filters, wave, rng=rng)

    # Compute photometry (add Gaussian noise)
    photometry_true = compute_photometry(true_spectrum, filters, wave)
    noise_level = 0.02 * photometry_true.std()
    photometry_noisy = photometry_true + rng.normal(scale=noise_level, size=n_filters)

    # Reconstruction
    W = construct_weight_matrix(filters, basis, wave)
    recon_coeffs = reconstruct_coefficients(photometry_noisy, W, alpha=1.0)
    recon_spectrum = synthesize_spectrum(recon_coeffs, basis)

    # Evaluation
    mse = np.mean((true_spectrum - recon_spectrum)**2)
    print(f"Mean squared error of reconstruction: {mse:.4e}")

if __name__ == "__main__":
    main()