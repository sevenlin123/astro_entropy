import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def create_wavelength_grid(n_points=1000, wl_min=400.0, wl_max=800.0):
    """Create a common wavelength grid in nanometers."""
    return np.linspace(wl_min, wl_max, n_points)

def normalize_transmission(trans):
    """Normalize filter transmission to have unit area."""
    area = simps(trans, axis=0)
    return trans / area

# ------------------------------------------------------------------
# Spectral model generation
# ------------------------------------------------------------------
def generate_basis_spectra(n_basis, wavelengths):
    """
    Generate a set of basis spectra (templates).
    Each basis spectrum is a smooth random curve.
    """
    rng = np.random.default_rng()
    basis = []
    for _ in range(n_basis):
        # Random coefficients for a low-order polynomial
        coeffs = rng.uniform(-1, 1, size=5)
        poly = sum(c * wavelengths**i for i, c in enumerate(coeffs))
        # Ensure positivity
        spec = np.abs(poly)
        # Smooth with a Gaussian kernel
        spec = gaussian_smooth(spec, sigma=5.0, wl_step=np.mean(np.diff(wavelengths)))
        basis.append(spec)
    return np.array(basis)  # shape (n_basis, n_wave)

def gaussian_smooth(signal, sigma, wl_step):
    """Smooth a signal with a Gaussian kernel."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(signal, sigma=sigma / wl_step)

def synthesize_star(coeffs, basis_spectra):
    """
    Combine basis spectra with given coefficients to produce a synthetic spectrum.
    coeffs: array of shape (n_basis,)
    """
    return np.dot(coeffs, basis_spectra)

# ------------------------------------------------------------------
# Filter generation
# ------------------------------------------------------------------
def generate_filters(n_filters, wavelengths):
    """
    Create a set of simple top-hat filter transmissions.
    Each filter covers a different wavelength range.
    """
    rng = np.random.default_rng()
    filters = []
    wl_min = wavelengths[0]
    wl_max = wavelengths[-1]
    band_width = (wl_max - wl_min) / (n_filters + 1)
    for i in range(n_filters):
        center = wl_min + (i + 1) * band_width
        lower = center - band_width / 2
        upper = center + band_width / 2
        trans = np.where((wavelengths >= lower) & (wavelengths <= upper), 1.0, 0.0)
        filters.append(normalize_transmission(trans))
    return np.array(filters)  # shape (n_filters, n_wave)

def compute_photometry(spectrum, filters):
    """
    Integrate the spectrum over each filter transmission.
    Returns fluxes proportional to observed photometric fluxes.
    """
    fluxes = []
    for filt in filters:
        flux = simps(spectrum * filt, x=wavelengths)
        fluxes.append(flux)
    return np.array(fluxes)

# ------------------------------------------------------------------
# Reconstruction
# ------------------------------------------------------------------
def build_design_matrix(photometry, filters, basis_spectra):
    """
    Construct the design matrix A such that:
        photometry = A @ coefficients
    where each row corresponds to a filter response integrated
    against each basis spectrum.
    """
    n_filters = filters.shape[0]
    n_basis = basis_spectra.shape[0]
    A = np.zeros((n_filters, n_basis))
    for i, filt in enumerate(filters):
        for j, basis in enumerate(basis_spectra):
            A[i, j] = simps(basis * filt, x=wavelengths)
    return A

def reconstruct_spectrum(measured_fluxes, filters, basis_spectra):
    """
    Estimate the coefficients of the basis spectra from photometric
    fluxes using linear least squares, then reconstruct the spectrum.
    """
    A = build_design_matrix(measured_fluxes, filters, basis_spectra)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(A, measured_fluxes)
    coeffs_est = reg.coef_
    reconstructed = synthesize_star(coeffs_est, basis_spectra)
    return coeffs_est, reconstructed

# ------------------------------------------------------------------
# Main routine: synthetic data generation and reconstruction
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Common wavelength grid
    wavelengths = create_wavelength_grid(n_points=1200)

    # 2. Generate basis spectra
    n_basis = 6
    basis_spectra = generate_basis_spectra(n_basis, wavelengths)

    # 3. Generate filter set
    n_filters = 5
    filters = generate_filters(n_filters, wavelengths)

    # 4. Create a random star
    rng = np.random.default_rng(seed=42)
    true_coeffs = rng.uniform(-1, 1, size=n_basis)
    true_spectrum = synthesize_star(true_coeffs, basis_spectra)

    # 5. Compute synthetic photometry
    photometry = compute_photometry(true_spectrum, filters)

    # 6. Reconstruct the spectrum from photometry
    coeffs_rec, spectrum_rec = reconstruct_spectrum(photometry, filters, basis_spectra)

    # 7. Compare results
    print("True coefficients:\n", true_coeffs)
    print("\nRecovered coefficients:\n", coeffs_rec)
    print("\nReconstruction error (norm):", np.linalg.norm(true_spectrum - spectrum_rec))

    # Optional: demonstrate that reconstruction matches photometry
    rec_photometry = compute_photometry(spectrum_rec, filters)
    print("\nPhotometry difference norm:", np.linalg.norm(photometry - rec_photometry))