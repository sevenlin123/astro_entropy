import numpy as np
from sklearn.linear_model import Ridge

def create_basis(n_pixels, n_basis):
    """Generate an orthonormal basis (e.g., Fourier modes) for spectra."""
    x = np.linspace(0, 1, n_pixels)
    basis = np.zeros((n_pixels, n_basis))
    for k in range(n_basis):
        if k == 0:
            basis[:, k] = np.ones(n_pixels)
        else:
            basis[:, k] = np.sqrt(2) * np.sin(np.pi * k * x)
    # Orthonormalize
    Q, _ = np.linalg.qr(basis)
    return Q

def generate_synthetic_spectra(n_samples, basis, noise_level=0.01, rng=None):
    """Generate synthetic spectra as random linear combinations of basis."""
    rng = np.random.default_rng(rng)
    coeffs = rng.normal(size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T
    spectra += noise_level * rng.standard_normal(spectra.shape)
    return spectra, coeffs

def create_filters(n_filters, n_pixels, rng=None):
    """Create simple top-hat filter transmission curves."""
    rng = np.random.default_rng(rng)
    filters = np.zeros((n_filters, n_pixels))
    for i in range(n_filters):
        start = rng.integers(0, n_pixels // 2)
        width = rng.integers(n_pixels // 8, n_pixels // 4)
        end = min(start + width, n_pixels)
        filters[i, start:end] = 1.0
    return filters

def compute_photometry(spectra, filters):
    """Integrate spectra over filter responses."""
    return spectra @ filters.T

def reconstruct_spectrum(photometry_sample, filters, basis, alpha=1.0):
    """Reconstruct a single spectrum from its photometry using ridge regression."""
    # Solve for coefficients: photometry = (basis @ filters.T) @ coeffs
    G = basis @ filters.T
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(G, photometry_sample)
    coeffs = reg.coef_
    spectrum = coeffs @ basis.T
    return spectrum, coeffs

def main():
    rng_seed = 42
    n_pixels = 100
    n_basis = 10
    n_samples = 50
    n_filters = 5

    rng = np.random.default_rng(rng_seed)

    basis = create_basis(n_pixels, n_basis)
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis, rng=rng)
    filters = create_filters(n_filters, n_pixels, rng=rng)
    photometry = compute_photometry(spectra, filters)

    # Reconstruct the first spectrum
    idx = 0
    recon_spec, recon_coeffs = reconstruct_spectrum(
        photometry[idx], filters, basis, alpha=0.1
    )

    print("True spectrum (first 10 values):", spectra[idx, :10])
    print("Reconstructed spectrum (first 10 values):", recon_spec[:10])
    print("True coefficients:", true_coeffs[idx])
    print("Reconstructed coefficients:", recon_coeffs)

if __name__ == "__main__":
    main()