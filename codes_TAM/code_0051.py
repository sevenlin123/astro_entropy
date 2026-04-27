import numpy as np
from scipy.stats import norm

def build_basis(wavelengths, n_basis=10):
    """Construct a set of Gaussian basis functions over the wavelength grid."""
    centers = np.linspace(wavelengths[0], wavelengths[-1], n_basis)
    widths = (wavelengths[-1] - wavelengths[0]) / (n_basis * 2)
    basis = np.array([norm.pdf(wavelengths, loc=c, scale=widths) for c in centers])
    return basis  # shape (n_basis, n_wavelengths)

def generate_filters(wavelengths, n_filters=5, fwhm=50.0):
    """Generate simple Gaussian filter curves."""
    filt_centers = np.linspace(450, 750, n_filters)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    filters = np.array([norm.pdf(wavelengths, loc=c, scale=sigma) for c in filt_centers])
    return filters  # shape (n_filters, n_wavelengths)

def compute_integrals(basis, filters):
    """Precompute integrals of basis functions through each filter."""
    # integral of basis * filter over wavelength, divided by integral of filter
    filt_norm = np.sum(filters, axis=1, keepdims=True)
    integrals = (filters[:, None, :] * basis[None, :, :]).sum(axis=2) / filt_norm
    return integrals  # shape (n_filters, n_basis)

def generate_synthetic_spectra(n_samples, basis, rng=None):
    """Generate synthetic spectra as random linear combinations of basis functions."""
    rng = rng if rng is not None else np.random.default_rng()
    coeffs = rng.standard_normal((n_samples, basis.shape[0]))
    spectra = coeffs @ basis
    return spectra, coeffs  # spectra shape (n_samples, n_wavelengths), coeffs shape (n_samples, n_basis)

def compute_photometry(spectra, filters):
    """Integrate spectra through each filter to produce photometric fluxes."""
    filt_norm = np.sum(filters, axis=1, keepdims=True)
    fluxes = (spectra[:, None, :] * filters[None, :, :]).sum(axis=2) / filt_norm
    return fluxes  # shape (n_samples, n_filters)

def reconstruct_coeffs(fluxes, integrals):
    """Reconstruct basis coefficients from photometric fluxes via least squares."""
    coeffs_rec, *_ = np.linalg.lstsq(integrals, fluxes.T, rcond=None)
    return coeffs_rec.T  # shape (n_samples, n_basis)

def reconstruct_spectrum(coeffs_rec, basis):
    """Reconstruct spectra from recovered coefficients."""
    return coeffs_rec @ basis

# Main routine
if __name__ == "__main__":
    # Define wavelength grid
    wav = np.linspace(400.0, 800.0, 1000)  # nm

    # Build spectral basis and filter set
    basis = build_basis(wav, n_basis=10)
    filters = generate_filters(wav, n_filters=5, fwhm=50.0)

    # Precompute integrals for reconstruction
    integrals = compute_integrals(basis, filters)

    # Generate synthetic dataset
    rng = np.random.default_rng(seed=42)
    n_samples = 20
    spectra_true, coeffs_true = generate_synthetic_spectra(n_samples, basis, rng=rng)

    # Compute photometric measurements
    fluxes = compute_photometry(spectra_true, filters)

    # Reconstruct spectra
    coeffs_rec = reconstruct_coeffs(fluxes, integrals)
    spectra_rec = reconstruct_spectrum(coeffs_rec, basis)

    # Evaluate reconstruction error
    mse = ((spectra_true - spectra_rec) ** 2).mean(axis=1)
    print("Mean squared reconstruction error per sample:", mse)