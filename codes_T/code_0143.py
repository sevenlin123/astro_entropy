import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def create_wavelength_grid(n_points, lam_min, lam_max):
    """Create a uniformly spaced wavelength grid."""
    return np.linspace(lam_min, lam_max, n_points)

def create_basis_spectra(n_basis, n_wave):
    """
    Generate synthetic basis spectra.
    Each basis spectrum is a sinusoid with random frequency and phase.
    """
    rng = np.random.default_rng(42)
    x = np.arange(n_wave)
    basis = []
    for _ in range(n_basis):
        freq = rng.uniform(0.01, 0.05)
        phase = rng.uniform(0, 2*np.pi)
        amp = rng.uniform(0.5, 1.5)
        spec = amp * np.sin(2*np.pi*freq*x + phase) + 1.0
        basis.append(spec)
    return np.vstack(basis)  # shape (n_basis, n_wave)

def create_filter_curves(n_filters, n_wave, lam_min, lam_max):
    """
    Create simple Gaussian filter transmission curves.
    """
    rng = np.random.default_rng(24)
    wavelengths = np.linspace(lam_min, lam_max, n_wave)
    filters = []
    centers = rng.uniform(lam_min+0.1*(lam_max-lam_min), lam_max-0.1*(lam_max-lam_min), size=n_filters)
    widths = rng.uniform(0.05*(lam_max-lam_min), 0.15*(lam_max-lam_min), size=n_filters)
    for c,w in zip(centers,widths):
        trans = np.exp(-0.5*((wavelengths-c)/w)**2)
        filters.append(trans)
    return np.vstack(filters)  # shape (n_filters, n_wave)

def generate_synthetic_spectrum(basis, coeffs, noise_std=0.0):
    """
    Combine basis spectra with given coefficients and optional Gaussian noise.
    """
    spectrum = np.dot(coeffs, basis)  # shape (n_wave,)
    if noise_std > 0:
        rng = np.random.default_rng()
        spectrum += rng.normal(scale=noise_std, size=spectrum.shape)
    return spectrum

def compute_photometry(spectrum, filters, wavelengths):
    """
    Integrate spectrum through each filter to obtain synthetic photometric fluxes.
    """
    # Ensure arrays are 1-D
    spectrum = np.asarray(spectrum)
    filters = np.asarray(filters)
    # Interpolate spectrum onto filter wavelengths if needed (here wavelengths match)
    integrals = np.trapz(filters * spectrum, wavelengths, axis=-1)
    return integrals  # shape (n_filters,)

def reconstruct_spectrum_from_photometry(filters, photometry, basis, wavelengths):
    """
    Reconstruct spectrum coefficients from photometry using linear least-squares.
    """
    # Build system matrix G where G[i,j] = ∫ filter_i * basis_j
    G = np.array([np.trapz(filters[i] * basis[j], wavelengths) for i in range(filters.shape[0]) 
                  for j in range(basis.shape[0])]).reshape(filters.shape[0], basis.shape[0])
    # Fit linear regression without intercept
    reg = LinearRegression(fit_intercept=False)
    reg.fit(G, photometry)
    coeffs = reg.coef_
    # Reconstruct spectrum
    reconstructed = np.dot(coeffs, basis)
    return reconstructed, coeffs

def main():
    # Parameters
    n_wave = 500
    n_basis = 5
    n_filters = 4
    lam_min, lam_max = 400.0, 800.0  # nm

    # Wavelength grid
    wavelengths = create_wavelength_grid(n_wave, lam_min, lam_max)

    # Basis spectra
    basis = create_basis_spectra(n_basis, n_wave)  # shape (n_basis, n_wave)

    # True coefficients for synthetic spectrum
    rng = np.random.default_rng(7)
    true_coeffs = rng.uniform(0.5, 1.5, size=n_basis)

    # Generate synthetic spectrum
    spectrum_true = generate_synthetic_spectrum(basis, true_coeffs, noise_std=0.02)

    # Filter curves
    filters = create_filter_curves(n_filters, n_wave, lam_min, lam_max)  # shape (n_filters, n_wave)

    # Compute synthetic photometry
    photometry = compute_photometry(spectrum_true, filters, wavelengths)

    # Reconstruct spectrum from photometry
    spectrum_rec, coeffs_est = reconstruct_spectrum_from_photometry(filters, photometry, basis, wavelengths)

    # Print results
    print("True coefficients   :", true_coeffs)
    print("Estimated coefficients:", coeffs_est)
    print("\nSpectrum reconstruction error (RMSE):",
          np.sqrt(np.mean((spectrum_true - spectrum_rec)**2)))

if __name__ == "__main__":
    main()