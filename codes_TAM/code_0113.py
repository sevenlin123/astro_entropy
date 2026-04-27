import numpy as np
from sklearn.linear_model import Ridge

# ---- 1. Spectral model ----------------------------------------------------
def create_wavelength_grid(start=3000, stop=10000, num=1000):
    """Create a wavelength grid in Ångstroms."""
    return np.linspace(start, stop, num)

def gaussian_template(wavelength, center, width):
    """Single Gaussian template."""
    return np.exp(-0.5 * ((wavelength - center) / width)**2)

def create_basis_functions(wavelength, n_bases=10):
    """Generate a set of Gaussian basis functions."""
    centers = np.linspace(wavelength.min(), wavelength.max(), n_bases)
    widths = np.full(n_bases, (wavelength.max() - wavelength.min()) / (n_bases * 4))
    basis = np.vstack([gaussian_template(wavelength, c, w) for c, w in zip(centers, widths)])
    return basis.T  # shape (len(wavelength), n_bases)

# ---- 2. Generate synthetic spectra ---------------------------------------
def generate_synthetic_spectrum(basis, coeff_mean=1.0, coeff_std=0.5):
    """Random linear combination of basis functions."""
    coeffs = np.random.normal(coeff_mean, coeff_std, size=basis.shape[1])
    spectrum = basis @ coeffs
    return spectrum, coeffs

# ---- 3. Generate photometric data -----------------------------------------
def create_filters(num_filters=5, wavelength=None):
    """Simple Gaussian filter transmission curves."""
    filt_centers = np.linspace(wavelength.min()+200, wavelength.max()-200, num_filters)
    filt_widths = np.full(num_filters, (wavelength.max() - wavelength.min()) / (num_filters * 3))
    filters = np.vstack([gaussian_template(wavelength, c, w) for c, w in zip(filt_centers, filt_widths)])
    return filters.T  # shape (len(wavelength), num_filters)

def compute_photometry(spectrum, filters):
    """Integrate spectrum through each filter."""
    # Normalize filter integrals to avoid scaling issues
    norms = np.trapz(filters, axis=0)
    fluxes = np.trapz(spectrum[:, None] * filters, axis=0) / norms
    return fluxes

# ---- 4. Reconstruction -----------------------------------------------------
def reconstruct_coefficients(filters, observed_fluxes, n_features, alpha=1.0):
    """Estimate spectrum coefficients from photometry using Ridge regression."""
    # Build design matrix: expected flux per unit coefficient
    # Each filter integrates each basis function
    design_matrix = np.trapz(filters, axis=0)  # shape (len(basis), n_features)
    # Solve for coefficients
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(design_matrix.T, observed_fluxes)
    return reg.coef_

def reconstruct_spectrum(basis, coeffs):
    """Rebuild spectrum from estimated coefficients."""
    return basis @ coeffs

# ---- 5. Demo ----------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Wavelength grid
    wav = create_wavelength_grid()

    # Basis functions
    basis = create_basis_functions(wav)

    # Synthetic spectrum
    true_spec, true_coeffs = generate_synthetic_spectrum(basis)

    # Filters
    filters = create_filters(num_filters=6, wavelength=wav)

    # Observed photometry (add Gaussian noise)
    true_fluxes = compute_photometry(true_spec, filters)
    noisy_fluxes = true_fluxes + np.random.normal(0, 0.01, size=true_fluxes.shape)

    # Reconstruct coefficients
    rec_coeffs = reconstruct_coefficients(filters, noisy_fluxes, n_features=basis.shape[1])

    # Reconstructed spectrum
    rec_spec = reconstruct_spectrum(basis, rec_coeffs)

    # Evaluate
    rmse = np.sqrt(np.mean((true_spec - rec_spec)**2))
    print(f"RMSE between true and reconstructed spectrum: {rmse:.4f}")

    # Simple visual check (requires matplotlib if desired)
    try:
        import matplotlib.pyplot as plt
        plt.plot(wav, true_spec, label='True spectrum')
        plt.plot(wav, rec_spec, '--', label='Reconstructed spectrum')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')
        plt.legend()
        plt.show()
    except ImportError:
        pass