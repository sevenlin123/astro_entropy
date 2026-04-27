import numpy as np
from sklearn.linear_model import LinearRegression

def create_basis(num_basis=5, num_wavelengths=200, seed=0):
    """Generate a simple spectral basis (e.g., Gaussian bumps)."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400, 800, num_wavelengths)
    bases = []
    for i in range(num_basis):
        center = rng.uniform(420, 780)
        width = rng.uniform(20, 80)
        amplitude = rng.uniform(0.5, 1.5)
        gauss = amplitude * np.exp(-((wavelengths - center) ** 2) / (2 * width ** 2))
        bases.append(gauss)
    return np.vstack(bases)  # shape: (num_basis, num_wavelengths)

def create_filters(num_filters=3, num_wavelengths=200, seed=1):
    """Generate simple rectangular filter transmission curves."""
    rng = np.random.default_rng(seed)
    filters = []
    for _ in range(num_filters):
        start = rng.uniform(400, 700)
        end = start + rng.uniform(30, 100)
        trans = np.zeros(num_wavelengths)
        mask = (np.linspace(400, 800, num_wavelengths) >= start) & \
               (np.linspace(400, 800, num_wavelengths) <= end)
        trans[mask] = rng.uniform(0.7, 1.0)
        filters.append(trans)
    return np.vstack(filters)  # shape: (num_filters, num_wavelengths)

def generate_spectra(basis, n_samples=10, noise_std=0.02, seed=2):
    """Sample linear combinations of basis spectra."""
    rng = np.random.default_rng(seed)
    n_basis, n_wave = basis.shape
    coeffs = rng.normal(loc=1.0, scale=0.3, size=(n_samples, n_basis))
    spectra = coeffs @ basis  # shape: (n_samples, n_wave)
    spectra += rng.normal(scale=noise_std, size=spectra.shape)
    return spectra, coeffs

def compute_photometry(spectra, filters):
    """Integrate spectra through filter transmission curves."""
    # spectra: (n_samples, n_wave)
    # filters: (n_filters, n_wave)
    return spectra @ filters.T  # shape: (n_samples, n_filters)

def reconstruct_spectra(photometry, filters, basis):
    """
    Reconstruct spectra from photometry via linear least squares.
    Returns reconstructed spectra and fitted coefficients.
    """
    n_samples = photometry.shape[0]
    # Construct the forward model matrix M = F * B^T
    # For each filter f and basis b: integral(f * b)
    M = filters @ basis.T  # shape: (n_filters, n_basis)
    # Solve least squares for each sample
    lr = LinearRegression(fit_intercept=False)
    lr.fit(M.T, photometry.T)  # fit for all samples simultaneously
    coeffs_hat = lr.coef_.T  # shape: (n_samples, n_basis)
    spectra_hat = coeffs_hat @ basis  # shape: (n_samples, n_wave)
    return spectra_hat, coeffs_hat

def main():
    basis = create_basis()
    filters = create_filters()
    spectra_true, coeffs_true = generate_spectra(basis)
    photometry = compute_photometry(spectra_true, filters)
    spectra_rec, coeffs_rec = reconstruct_spectra(photometry, filters, basis)

    # Simple diagnostics
    recon_error = np.mean(np.abs(spectra_true - spectra_rec))
    coeff_error = np.mean(np.abs(coeffs_true - coeffs_rec))
    print(f"Mean absolute spectrum error: {recon_error:.4f}")
    print(f"Mean absolute coefficient error: {coeff_error:.4f}")

if __name__ == "__main__":
    main()