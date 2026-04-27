import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Define a simple spectral basis (e.g. Gaussian functions)
# ------------------------------------------------------------------
def spectral_basis(wavelengths, n_basis=5):
    """
    Create a matrix of basis function evaluations.
    Each column corresponds to one basis function evaluated at `wavelengths`.
    """
    xs = np.linspace(wavelengths.min(), wavelengths.max(), n_basis)
    widths = (wavelengths.max() - wavelengths.min()) / (3 * n_basis)
    basis = np.exp(-0.5 * ((wavelengths[:, None] - xs[None, :]) / widths) ** 2)
    return basis

# ------------------------------------------------------------------
# Generate synthetic spectra
# ------------------------------------------------------------------
def synthesize_spectrum(basis, coeffs):
    """
    Generate a spectrum as a linear combination of the given basis functions.
    """
    return basis @ coeffs

# ------------------------------------------------------------------
# Generate photometric observations from spectra
# ------------------------------------------------------------------
def create_filters(n_filters=3, wavelengths=np.linspace(400, 700, 301)):
    """
    Create simple Gaussian filters for photometric bands.
    """
    filters = []
    centers = np.linspace(wavelengths.min(), wavelengths.max(), n_filters)
    widths = (wavelengths.max() - wavelengths.min()) / (3 * n_filters)
    for c in centers:
        filt = np.exp(-0.5 * ((wavelengths - c) / widths) ** 2)
        filters.append(filt)
    return np.array(filters)

def photometry_from_spectrum(spectrum, filters, wavelengths):
    """
    Integrate spectrum over each filter transmission curve.
    """
    fluxes = []
    for filt in filters:
        flux = simps(spectrum * filt, wavelengths)
        fluxes.append(flux)
    return np.array(fluxes)

# ------------------------------------------------------------------
# Reconstruction of spectrum from photometry
# ------------------------------------------------------------------
def reconstruct_spectrum(photometry, filters, wavelengths, n_basis=5):
    """
    Reconstruct the spectrum by solving a least‑squares problem
    in the space spanned by the spectral basis.
    """
    basis = spectral_basis(wavelengths, n_basis)
    # Build design matrix: each row corresponds to a filter integrated over the basis
    G = np.array([simps(basis * filt[:, None], wavelengths) for filt in filters])
    # Fit coefficients using linear regression
    reg = LinearRegression(fit_intercept=False).fit(G.T, photometry)
    coeffs = reg.coef_
    # Reconstruct spectrum
    return synthesize_spectrum(basis, coeffs), coeffs

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wav = np.linspace(400, 700, 301)  # nm

    # Spectral basis
    n_basis = 5
    basis = spectral_basis(wav, n_basis)

    # Random coefficients for a synthetic target spectrum
    true_coeffs = np.random.randn(n_basis)
    target_spectrum = synthesize_spectrum(basis, true_coeffs)

    # Photometric filters
    n_filters = 3
    filters = create_filters(n_filters, wav)

    # Photometric observations
    photo = photometry_from_spectrum(target_spectrum, filters, wav)

    # Reconstruction
    recon_spectrum, recon_coeffs = reconstruct_spectrum(photo, filters, wav, n_basis)

    # Print results
    print("True coefficients :", true_coeffs)
    print("Reconstructed coeffs:", recon_coeffs)
    print("Spectral difference RMS :", np.sqrt(np.mean((target_spectrum - recon_spectrum)**2)))