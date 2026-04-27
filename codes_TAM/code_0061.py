import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

def build_basis(wavelength, n_basis):
    """Generate a set of Gaussian basis functions."""
    centers = np.linspace(wavelength.min(), wavelength.max(), n_basis)
    widths = (wavelength.max() - wavelength.min()) / (n_basis * 2.0)
    B = np.array([
        np.exp(-0.5 * ((wavelength - c) / widths)**2)
        for c in centers
    ]).T  # shape: (len(wavelength), n_basis)
    return B

def generate_synthetic_spectra(B, n_samples, noise_std=0.02):
    """Create synthetic spectra as random combinations of basis functions."""
    n_basis = B.shape[1]
    coeffs = np.random.randn(n_samples, n_basis)
    spectra = B @ coeffs.T + noise_std * np.random.randn(*B.shape)
    return spectra.T, coeffs  # shape: (len(wavelength), n_samples)

def gaussian_filter(wavelength, center, width):
    """Return a normalized Gaussian filter transmission curve."""
    filt = np.exp(-0.5 * ((wavelength - center) / width)**2)
    return filt / np.trapz(filt, wavelength)

def build_filters(wavelength, n_filters):
    """Construct several Gaussian bandpass filters."""
    centers = np.linspace(wavelength.min() + 50, wavelength.max() - 50, n_filters)
    width = (wavelength.max() - wavelength.min()) / (n_filters * 4.0)
    filters = [gaussian_filter(wavelength, c, width) for c in centers]
    return filters

def compute_photometry(spectra, filters, wavelength):
    """Integrate spectra over each filter to obtain synthetic photometric points."""
    phot = []
    for filt in filters:
        flux = simps(spectra * filt[:, None], wavelength, axis=0)
        phot.append(flux)
    return np.array(phot).T  # shape: (n_samples, n_filters)

def build_response_matrix(B, filters, wavelength):
    """Project basis functions through filters to get the response matrix."""
    n_samples = B.shape[0]  # using single spectrum length
    responses = []
    for filt in filters:
        resp = simps(B * filt[:, None], wavelength, axis=0)
        responses.append(resp)
    return np.vstack(responses).T  # shape: (n_basis, n_filters)

def reconstruct_spectrum(phot, B, filters, wavelength):
    """Reconstruct spectra from photometry using linear regression."""
    # Build response matrix R such that phot = R^T * coeffs
    R = build_response_matrix(B, filters, wavelength)
    # Solve least-squares for coefficients for each sample
    coeffs, *_ = np.linalg.lstsq(R.T, phot.T, rcond=None)
    coeffs = coeffs.T  # shape: (n_samples, n_basis)
    # Reconstruct spectra
    reconstructed = B @ coeffs.T
    return reconstructed

def main():
    # Define wavelength grid
    wavelength = np.linspace(3000, 25000, 2000)  # Ångströms

    # Build spectral basis
    n_basis = 20
    B = build_basis(wavelength, n_basis)

    # Generate synthetic spectra
    n_samples = 10
    spectra, true_coeffs = generate_synthetic_spectra(B, n_samples)

    # Build filters
    n_filters = 5
    filters = build_filters(wavelength, n_filters)

    # Compute synthetic photometry
    phot = compute_photometry(spectra, filters, wavelength)

    # Reconstruct spectra
    reconstructed = reconstruct_spectrum(phot, B, filters, wavelength)

    # Simple evaluation
    rmse = np.sqrt(np.mean((spectra - reconstructed)**2))
    print(f"Reconstruction RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()