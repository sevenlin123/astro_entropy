import numpy as np
from scipy.special import erf
from numpy.linalg import pinv

# ---------- Spectral model ----------
def gaussian(x, mu, sigma):
    """Gaussian profile."""
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def generate_basis_spectra(n_bases, wavelengths):
    """Generate a set of simple Gaussian basis spectra."""
    rng = np.random.default_rng()
    basis = []
    for _ in range(n_bases):
        mu = rng.uniform(wavelengths[0], wavelengths[-1])
        sigma = rng.uniform((wavelengths[-1] - wavelengths[0]) / 10,
                            (wavelengths[-1] - wavelengths[0]) / 4)
        amp = rng.uniform(0.5, 1.5)
        spec = amp * gaussian(wavelengths, mu, sigma)
        basis.append(spec)
    return np.vstack(basis)  # shape (n_bases, n_wave)

def generate_samples(basis, n_samples, coeff_range=(0.5, 1.5), noise=0.01):
    """Generate synthetic spectra as linear combinations of basis spectra."""
    rng = np.random.default_rng()
    n_bases, n_wave = basis.shape
    coeffs = rng.uniform(coeff_range[0], coeff_range[1],
                         size=(n_samples, n_bases))
    spectra = coeffs @ basis
    spectra += noise * rng.standard_normal(spectra.shape)
    return spectra, coeffs

# ---------- Photometry ----------
def generate_filters(n_filters, wavelengths):
    """Generate simple Gaussian filter transmission curves."""
    rng = np.random.default_rng()
    filters = []
    for _ in range(n_filters):
        center = rng.uniform(wavelengths[0], wavelengths[-1])
        width = rng.uniform((wavelengths[-1] - wavelengths[0]) / 20,
                            (wavelengths[-1] - wavelengths[0]) / 8)
        trans = gaussian(wavelengths, center, width)
        trans /= trans.max()  # normalise to unity peak
        filters.append(trans)
    return np.vstack(filters)  # shape (n_filters, n_wave)

def compute_photometry(spectra, filters):
    """Integrate spectra through filter transmission curves."""
    # spectra: (n_samples, n_wave)
    # filters: (n_filters, n_wave)
    return spectra @ filters.T  # shape (n_samples, n_filters)

# ---------- Reconstruction ----------
def reconstruct_coefficients(photometry, filter_integrals):
    """
    Recover mixture coefficients from photometry.

    filter_integrals has shape (n_bases, n_filters)
    """
    return photometry @ pinv(filter_integrals.T)

def reconstruct_spectra(coeffs, basis):
    """Reconstruct spectra from recovered coefficients."""
    return coeffs @ basis

# ---------- Main routine ----------
def main():
    # Wavelength grid (nm)
    wavelengths = np.linspace(400, 800, 1000)

    # Basis spectra
    basis = generate_basis_spectra(n_bases=5, wavelengths=wavelengths)

    # Synthetic sample spectra
    samples, true_coeffs = generate_samples(basis, n_samples=50, noise=0.02)

    # Filters
    filters = generate_filters(n_filters=10, wavelengths=wavelengths)

    # Photometry from synthetic spectra
    photometry = compute_photometry(samples, filters)

    # Precompute filter integrals of basis spectra
    filter_integrals = basis @ filters.T  # shape (n_bases, n_filters)

    # Reconstruct coefficients from photometry
    recon_coeffs = reconstruct_coefficients(photometry, filter_integrals)

    # Reconstruct spectra
    recon_spectra = reconstruct_spectra(recon_coeffs, basis)

    # Error metric
    rel_error = np.mean(np.abs(recon_spectra - samples) / np.maximum(1e-12, samples))
    print(f"Mean relative reconstruction error: {rel_error:.3f}")

if __name__ == "__main__":
    main()