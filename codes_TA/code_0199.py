import numpy as np
from scipy.signal import convolve
from sklearn.linear_model import Ridge

def generate_wavelength_grid(n_wave, lam_min=350, lam_max=950):
    return np.linspace(lam_min, lam_max, n_wave)

def gaussian_basis_functions(wavelengths, n_bases):
    """Generate a set of Gaussian basis functions."""
    gaussians = []
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_bases)
    widths = (wavelengths[-1] - wavelengths[0]) / (n_bases * 2)
    for c in centers:
        g = np.exp(-0.5 * ((wavelengths - c) / widths)**2)
        gaussians.append(g)
    return np.vstack(gaussians).T  # shape (n_wave, n_bases)

def generate_synthetic_spectra(basis, n_samples, noise_std=0.01, random_state=None):
    rng = np.random.default_rng(random_state)
    coeffs = rng.standard_normal((n_samples, basis.shape[1]))
    spectra = basis @ coeffs.T  # (n_wave, n_samples)
    spectra += noise_std * rng.standard_normal(spectra.shape)
    return spectra.T, coeffs  # (n_samples, n_wave), (n_samples, n_bases)

def generate_filters(wavelengths, n_filters, random_state=None):
    rng = np.random.default_rng(random_state)
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform(10, 100)
        filt = np.exp(-0.5 * ((wavelengths - center) / width)**2)
        filt /= filt.sum()  # normalize
        filters.append(filt)
    return np.vstack(filters)  # (n_filters, n_wave)

def compute_photometry(spectra, filters):
    """Integrate spectra over filter responses."""
    return spectra @ filters.T  # (n_samples, n_filters)

def reconstruct_spectrum(filters, basis, photometry, alpha=1.0):
    """Reconstruct spectra using Ridge regression on filter space."""
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(filters, photometry.T)
    coeffs_est = reg.coef_.T  # (n_samples, n_filters)
    # Project back onto basis functions
    return coeffs_est @ basis.T  # (n_samples, n_wave)

def main():
    np.set_printoptions(precision=3, suppress=True)
    n_wave = 200
    n_bases = 6
    n_samples = 30
    n_filters = 4

    wavelengths = generate_wavelength_grid(n_wave)
    basis = gaussian_basis_functions(wavelengths, n_bases)
    spectra, true_coeffs = generate_synthetic_spectra(basis, n_samples, random_state=42)
    filters = generate_filters(wavelengths, n_filters, random_state=24)
    photometry = compute_photometry(spectra, filters)

    reconstructed = reconstruct_spectrum(filters, basis, photometry, alpha=0.1)

    print("True spectra shape:", spectra.shape)
    print("Photometry shape:", photometry.shape)
    print("Reconstructed spectra shape:", reconstructed.shape)
    # Simple comparison for first sample
    idx = 0
    print("\nSample", idx)
    print("Original spectrum:", spectra[idx][:5])
    print("Reconstructed spectrum:", reconstructed[idx][:5])

if __name__ == "__main__":
    main()