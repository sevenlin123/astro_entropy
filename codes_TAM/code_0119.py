import numpy as np
from scipy.special import erf
from sklearn.linear_model import LinearRegression

def gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def generate_spectral_model(wavelengths, n_components=5, rng=None):
    """Generate basis spectra as Gaussian components."""
    rng = np.random.default_rng(rng)
    mus = rng.uniform(4000, 8000, size=n_components)
    sigmas = rng.uniform(50, 200, size=n_components)
    amps = rng.uniform(0.5, 1.5, size=n_components)
    basis = np.vstack([amps[i] * gaussian(wavelengths, mus[i], sigmas[i])
                       for i in range(n_components)])
    return basis

def synthesize_spectra(basis, n_spectra=10, rng=None):
    """Create synthetic spectra as linear combinations of basis."""
    rng = np.random.default_rng(rng)
    coeffs = rng.uniform(0.2, 1.0, size=(n_spectra, basis.shape[0]))
    spectra = coeffs @ basis
    return spectra, coeffs

def generate_filters(wavelengths, n_filters=3, rng=None):
    """Generate simple Gaussian filters."""
    rng = np.random.default_rng(rng)
    mus = rng.uniform(4500, 7500, size=n_filters)
    sigmas = rng.uniform(100, 300, size=n_filters)
    filters = np.vstack([gaussian(wavelengths, mus[i], sigmas[i])
                         for i in range(n_filters)])
    return filters

def compute_photometry(spectra, filters):
    """Integrate spectra over filters to obtain photometric fluxes."""
    # Simple trapezoidal integration
    fluxes = spectra @ filters.T
    return fluxes

def reconstruct_spectrum(photometry, filters, wavelengths):
    """Reconstruct spectra using linear regression on filter responses."""
    model = LinearRegression(fit_intercept=False)
    model.fit(filters.T, np.eye(filters.shape[0]))
    recon_basis = model.coef_.T  # shape (n_wavelengths, n_filters)
    reconstructed = photometry @ recon_basis
    return reconstructed

def main():
    rng = 42
    wavelengths = np.linspace(3500, 9000, 500)  # nm
    basis = generate_spectral_model(wavelengths, n_components=7, rng=rng)
    spectra, true_coeffs = synthesize_spectra(basis, n_spectra=15, rng=rng)
    filters = generate_filters(wavelengths, n_filters=4, rng=rng)
    photometry = compute_photometry(spectra, filters)
    recon_spectra = reconstruct_spectrum(photometry, filters, wavelengths)

    print("Synthetic spectra shape:", spectra.shape)
    print("Photometry shape:", photometry.shape)
    print("Reconstructed spectra shape:", recon_spectra.shape)

if __name__ == "__main__":
    main()