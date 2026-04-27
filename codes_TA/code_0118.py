import numpy as np
from scipy.special import erfc
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model: linear combination of basis spectra
# ----------------------------------------------------------------------
def create_basis_spectra(n_wave=81, n_bases=3):
    """
    Create a set of simple basis spectra.
    Returns an array of shape (n_bases, n_wave).
    """
    wavelength = np.linspace(400, 800, n_wave)  # nm
    bases = []
    centers = [450, 550, 650]
    widths = [20, 30, 25]
    for c, w in zip(centers, widths):
        base = np.exp(-0.5 * ((wavelength - c) / w)**2)
        bases.append(base)
    return np.vstack(bases)

def generate_synthetic_spectra(n_samples, basis):
    """
    Generate synthetic spectra as random linear combinations of basis spectra.
    Returns an array of shape (n_samples, n_wave).
    """
    n_bases, n_wave = basis.shape
    coeffs = np.random.rand(n_samples, n_bases)
    spectra = coeffs @ basis  # (n_samples, n_wave)
    return spectra, coeffs

# ----------------------------------------------------------------------
# Photometric filter definition
# ----------------------------------------------------------------------
def gaussian_filter_response(wavelength, center, sigma):
    """Top-hat Gaussian filter response."""
    return np.exp(-0.5 * ((wavelength - center) / sigma)**2)

def create_filters(n_filters=3):
    """
    Create a set of simple Gaussian photometric filters.
    Returns a list of filter response arrays.
    """
    wavelength = np.linspace(400, 800, 81)  # same grid as spectra
    centers = [430, 530, 630]
    sigmas = [15, 20, 18]
    filters = [gaussian_filter_response(wavelength, c, s) for c, s in zip(centers, sigmas)]
    return filters

# ----------------------------------------------------------------------
# Forward modelling: spectra → photometry
# ----------------------------------------------------------------------
def compute_photometry(spectrum, filters):
    """
    Compute broadband photometry for a single spectrum.
    Returns an array of fluxes (one per filter).
    """
    phot = []
    for f in filters:
        flux = np.trapz(spectrum * f, x=np.linspace(400, 800, len(f))) / np.trapz(f, x=np.linspace(400, 800, len(f)))
        phot.append(flux)
    return np.array(phot)

# ----------------------------------------------------------------------
# Pre-compute basis photometry for regression
# ----------------------------------------------------------------------
def precompute_basis_photometry(basis, filters):
    """
    For each basis spectrum, compute its photometry through all filters.
    Returns a matrix of shape (n_filters, n_bases).
    """
    n_bases = basis.shape[0]
    n_filters = len(filters)
    phots = np.zeros((n_filters, n_bases))
    for i, b in enumerate(basis):
        phots[:, i] = compute_photometry(b, filters)
    return phots

# ----------------------------------------------------------------------
# Reconstruction: photometry → coefficients → spectrum
# ----------------------------------------------------------------------
def reconstruct_coefficients(photometry, basis_photometry, alpha=1e-4):
    """
    Solve for coefficients that best reproduce the observed photometry.
    Uses Ridge regression for numerical stability.
    """
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(basis_photometry.T, photometry)
    return ridge.coef_

def reconstruct_spectrum(coeffs, basis):
    """Reconstruct spectrum from coefficients and basis spectra."""
    return coeffs @ basis

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
def main():
    # Define wavelength grid
    wavelength = np.linspace(400, 800, 81)  # nm

    # Create basis spectra
    basis = create_basis_spectra(n_wave=len(wavelength), n_bases=3)

    # Generate synthetic spectra
    n_samples = 10
    spectra, true_coeffs = generate_synthetic_spectra(n_samples, basis)

    # Define photometric filters
    filters = create_filters(n_filters=3)

    # Pre-compute basis photometry
    basis_photometry = precompute_basis_photometry(basis, filters)

    # Pick a sample to reconstruct
    idx = 0
    sample_spec = spectra[idx]
    sample_coeff = true_coeffs[idx]

    # Compute photometry for this sample
    photometry = compute_photometry(sample_spec, filters)

    # Reconstruct coefficients
    recon_coeffs = reconstruct_coefficients(photometry, basis_photometry)

    # Reconstruct spectrum
    recon_spectrum = reconstruct_spectrum(recon_coeffs, basis)

    # Evaluate reconstruction
    error = np.linalg.norm(sample_spec - recon_spectrum) / np.linalg.norm(sample_spec)
    print(f"Reconstruction relative L2 error: {error:.4f}")

    # Show true vs reconstructed coefficients
    print("True coefficients:", sample_coeff)
    print("Reconstructed coeffs:", recon_coeffs)

if __name__ == "__main__":
    main()