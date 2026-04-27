import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Define spectral model (basis functions)
# ----------------------------------------------------------------------
def define_spectral_model(n_basis=10, wave_min=400, wave_max=800, n_wave=1000):
    """
    Create a set of Gaussian basis functions spanning the wavelength range.
    Returns:
        wavelengths : (n_wave,) array
        basis       : (n_wave, n_basis) array of basis values
    """
    wavelengths = np.linspace(wave_min, wave_max, n_wave)
    centers = np.linspace(wave_min, wave_max, n_basis)
    widths  = (wave_max - wave_min) / (2 * n_basis) * np.ones(n_basis)

    basis = np.exp(-0.5 * ((wavelengths[:, None] - centers[None, :]) / widths[None, :])**2)
    return wavelengths, basis

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_synthetic_spectra(basis, n_samples=50, noise_std=0.05):
    """
    Randomly combine basis functions to produce synthetic spectra.
    Adds Gaussian noise to each spectrum.
    Returns:
        spectra : (n_samples, n_wave) array
        coeffs  : (n_samples, n_basis) array of true coefficients
    """
    n_basis = basis.shape[1]
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = coeffs @ basis.T
    spectra += noise_std * np.random.randn(*spectra.shape)
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# ----------------------------------------------------------------------
def define_filters(n_filters=3, wave_min=400, wave_max=800, n_wave=1000):
    """
    Create simple Gaussian filter transmission curves.
    Returns:
        filters : (n_filters, n_wave) array
        filt_centers : array of filter center wavelengths
    """
    wavelengths = np.linspace(wave_min, wave_max, n_wave)
    filt_centers = np.linspace(wave_min + 0.25*(wave_max-wave_min),
                               wave_max - 0.25*(wave_max-wave_min),
                               n_filters)
    filt_widths = 20.0 * np.ones(n_filters)  # nm
    filters = np.exp(-0.5 * ((wavelengths[:, None] - filt_centers[None, :]) /
                            filt_widths[None, :])**2)
    return filters, filt_centers

def generate_photometric_data(spectra, filters, wavelengths):
    """
    Integrate each spectrum over each filter to get photometric fluxes.
    Uses Simpson's rule for numerical integration.
    Returns:
        photometry : (n_samples, n_filters) array of integrated fluxes
    """
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    photometry = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        # Element-wise multiplication then integrate over wavelength
        integrand = spectra * filters[i][None, :]
        photometry[:, i] = simps(integrand, wavelengths, axis=1)
    return photometry

# ----------------------------------------------------------------------
# 4. Reconstruct synthetic spectrum from photometry
# ----------------------------------------------------------------------
def reconstruct_spectrum_from_photometry(photometry, filters, wavelengths, n_basis=10):
    """
    Estimate spectrum coefficients by solving a linear system that maps
    coefficients to photometric fluxes via filter integrals.
    Returns:
        recon_coeffs : (n_samples, n_basis) array of estimated coefficients
        recon_spectra : (n_samples, n_wave) array of reconstructed spectra
    """
    # Compute design matrix A such that coeffs @ A.T ≈ photometry
    n_filters = filters.shape[0]
    n_wave = wavelengths.size
    A = np.zeros((n_basis, n_filters))
    for j in range(n_filters):
        # Integrate each basis function through filter j
        integrand = basis @ filters[j]
        A[:, j] = simps(integrand, wavelengths)
    # Transpose to shape (n_filters, n_basis)
    A_T = A.T

    # Solve for coefficients using linear regression (least squares)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A_T, photometry.T)
    recon_coeffs = reg.coef_.T

    # Reconstruct spectra from coefficients
    recon_spectra = recon_coeffs @ basis.T
    return recon_coeffs, recon_spectra

# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define spectral model
    wavelengths, basis = define_spectral_model(n_basis=10)

    # Generate synthetic spectra
    spectra, true_coeffs = generate_synthetic_spectra(basis, n_samples=50)

    # Define filters
    filters, filt_centers = define_filters(n_filters=3)

    # Generate photometric data
    photometry = generate_photometric_data(spectra, filters, wavelengths)

    # Reconstruct spectra from photometry
    recon_coeffs, recon_spectra = reconstruct_spectrum_from_photometry(
        photometry, filters, wavelengths, n_basis=10)

    # Simple accuracy check: mean squared error between true and reconstructed spectra
    mse = np.mean((spectra - recon_spectra)**2)
    print(f"Mean squared error of reconstruction: {mse:.4f}")

    # Optionally, print a few spectra for visual inspection
    import matplotlib.pyplot as plt
    idx = 0  # first spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectra[idx], label="True Spectrum")
    plt.plot(wavelengths, recon_spectra[idx], '--', label="Reconstructed Spectrum")
    plt.title("Spectrum Reconstruction Example")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux")
    plt.legend()
    plt.tight_layout()
    plt.show()