import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# 1. Spectral model – basis functions
# ------------------------------------------------------------------
def create_spectral_basis(n_basis=5, n_wave=200, wave_start=400, wave_end=800):
    """
    Generates a set of Gaussian basis spectra.
    """
    wavelengths = np.linspace(wave_start, wave_end, n_wave)
    centers = np.linspace(wave_start + 0.2*(wave_end-wave_start),
                          wave_end - 0.2*(wave_end-wave_start), n_basis)
    widths = 0.05 * (wave_end - wave_start)  # fixed width
    basis = []
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c)/widths)**2)
        basis.append(g)
    return np.array(basis)  # shape (n_basis, n_wave)

# ------------------------------------------------------------------
# 2. Synthetic spectra generation
# ------------------------------------------------------------------
def generate_synthetic_spectra(n_samples, basis, rng=None):
    """
    Creates synthetic spectra as random linear combinations of basis.
    """
    if rng is None:
        rng = np.random.default_rng()
    coeffs = rng.uniform(low=0.5, high=1.5, size=(n_samples, basis.shape[0]))
    spectra = coeffs @ basis  # (n_samples, n_wave)
    return spectra, coeffs

# ------------------------------------------------------------------
# 3. Photometric filter construction
# ------------------------------------------------------------------
def create_filters(n_filters=4, n_wave=200, wave_start=400, wave_end=800):
    """
    Generates simple Gaussian filter transmission curves.
    """
    wavelengths = np.linspace(wave_start, wave_end, n_wave)
    centers = np.linspace(wave_start + 0.15*(wave_end-wave_start),
                          wave_end - 0.15*(wave_end-wave_start), n_filters)
    widths = 0.08 * (wave_end - wave_start)
    filters = []
    for c in centers:
        f = np.exp(-0.5 * ((wavelengths - c)/widths)**2)
        filters.append(f)
    return np.array(filters)  # shape (n_filters, n_wave)

# ------------------------------------------------------------------
# 4. Compute photometry from spectra
# ------------------------------------------------------------------
def compute_photometry(spectra, filters):
    """
    Integrates spectra through filter transmittances.
    """
    # Simple trapezoidal integration
    phot = spectra @ filters.T  # (n_samples, n_filters)
    return phot

# ------------------------------------------------------------------
# 5. Reconstruction framework
# ------------------------------------------------------------------
def compute_filter_responses(basis, filters):
    """
    Computes the response of each basis function to each filter.
    """
    n_basis, n_wave = basis.shape
    n_filters = filters.shape[0]
    responses = np.zeros((n_filters, n_basis))
    for i in range(n_filters):
        for j in range(n_basis):
            responses[i, j] = np.trapz(basis[j] * filters[i], axis=-1)
    return responses  # shape (n_filters, n_basis)

def reconstruct_spectra(photometry, basis, filters):
    """
    Reconstructs spectra coefficients from photometry using linear regression.
    """
    # Build design matrix: responses of basis to filters
    A = compute_filter_responses(basis, filters)  # (n_filters, n_basis)
    # Fit coefficients for each sample
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A.T, photometry.T)  # shape: (n_basis, n_samples)
    coeffs_recon = reg.coef_.T      # (n_samples, n_basis)
    # Reconstruct spectra
    spectra_recon = coeffs_recon @ basis
    return spectra_recon, coeffs_recon

# ------------------------------------------------------------------
# Main routine – synthetic data pipeline
# ------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # Parameters
    n_samples = 20
    n_basis = 5
    n_wave = 200
    n_filters = 4

    # 1. Create basis
    basis = create_spectral_basis(n_basis=n_basis, n_wave=n_wave)

    # 2. Generate synthetic spectra
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis, rng=rng)

    # 3. Create filters
    filters = create_filters(n_filters=n_filters, n_wave=n_wave)

    # 4. Compute photometry
    photometry = compute_photometry(spectra_true, filters)

    # 5. Reconstruct spectra
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, basis, filters)

    # Display simple error metrics
    mse = np.mean((spectra_true - spectra_rec)**2)
    coeff_error = np.mean((coeffs_true - coeffs_rec)**2)
    print(f"Mean squared error of spectrum reconstruction: {mse:.4f}")
    print(f"Mean squared error of coefficient reconstruction: {coeff_error:.4f}")