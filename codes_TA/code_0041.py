import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# 1) Spectral model: simple linear combination of basis spectra
# ------------------------------------------------------------

def build_basis(n_wave=100, n_basis=5):
    """
    Generate synthetic basis spectra: sinusoidal components with random phases
    """
    wavelengths = np.linspace(400, 2500, n_wave)
    basis = []
    for i in range(n_basis):
        freq = np.random.uniform(0.01, 0.05)
        phase = np.random.uniform(0, 2*np.pi)
        spec = np.sin(freq * wavelengths + phase)
        basis.append(spec)
    return wavelengths, np.column_stack(basis)

# ------------------------------------------------------------
# 2) Synthetic spectra generation
# ------------------------------------------------------------

def generate_synthetic_spectrum(wavelengths, basis, coeffs):
    """
    Linear combination of basis spectra
    """
    return np.dot(basis.T, coeffs)

def create_synthetic_dataset(num_samples=50, n_wave=100, n_basis=5):
    """Generate dataset of spectra and their coefficients"""
    wavelengths, basis = build_basis(n_wave=n_wave, n_basis=n_basis)
    coeffs_list = []
    spectra = []
    for _ in range(num_samples):
        coeffs = np.random.randn(n_basis)
        spec = generate_synthetic_spectrum(wavelengths, basis, coeffs)
        spectra.append(spec)
        coeffs_list.append(coeffs)
    return wavelengths, basis, np.array(spectra), np.array(coeffs_list)

# ------------------------------------------------------------
# 3) Photometry synthesis
# ------------------------------------------------------------

def compute_filter_response(wavelengths, center, width):
    """
    Gaussian filter response function
    """
    return np.exp(-0.5*((wavelengths - center)/width)**2)

def convert_to_photometry(spectrum, wavelengths, filter_centers, filter_widths):
    """
    Convert each spectrum to synthetic photometric measurements
    using Gaussian filters.
    """
    photom = []
    for center, width in zip(filter_centers, filter_widths):
        response = compute_filter_response(wavelengths, center, width)
        flux = np.trapz(spectrum * response, wavelengths)
        photom.append(flux)
    return np.array(photom)

# ------------------------------------------------------------
# 4) Spectrum reconstruction from photometry
# ------------------------------------------------------------

def reconstruct_from_photometry(photometry, basis, wavelengths,
                                 filter_centers, filter_widths, n_iter=500):
    """
    Recover spectral coefficients by solving a non‑linear least squares
    optimization problem with constraint that the coefficients
    are bounded by ±10.
    The function returns the reconstructed spectrum.
    """
    # Precompute response matrix
    response_matrix = []
    for p, width in zip(filter_centers, filter_widths):
        response = compute_filter_response(wavelengths, p, width)
        response_matrix.append(np.dot(basis.T, response))
            #   *   ... 
        if i < ???   # ???????...
       ???...?