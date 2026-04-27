import numpy as np
from scipy.integrate import simps
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# 1. Define spectral model (basis functions)
# ----------------------------------------------------------------------
def create_basis_spectra(n_basis, wavelength):
    """Return an array of basis spectra (gaussian bumps)."""
    basis = []
    centers = np.linspace(wavelength.min(), wavelength.max(), n_basis)
    widths = (wavelength.max() - wavelength.min()) / (2 * n_basis)
    for c in centers:
        basis.append(norm.pdf(wavelength, loc=c, scale=widths))
    return np.vstack(basis)  # shape (n_basis, n_wave)

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra
# ----------------------------------------------------------------------
def generate_true_spectrum(coeffs, basis_spectra):
    """Linear combination of basis spectra."""
    return coeffs @ basis_spectra

# ----------------------------------------------------------------------
# 3. Generate photometric data from synthetic spectra
# ----------------------------------------------------------------------
def create_filter_response(center, width, wavelength):
    """Gaussian filter transmission."""
    return norm.pdf(wavelength, loc=center, scale=width)

def compute_flux_in_band(spectrum, filt, wavelength):
    """Flux through a single filter."""
    numerator = simps(spectrum * filt, wavelength)
    denom = simps(filt, wavelength)
    return numerator / denom

def generate_photometry(spectrum, filters, wavelength):
    """Compute photometry for all filters."""
    return np.array([compute_flux_in_band(spectrum, f, wavelength) for f in filters])

# ----------------------------------------------------------------------
# 4. Reconstruction framework
# ----------------------------------------------------------------------
def build_filter_matrix(filters, basis_spectra, wavelength):
    """Matrix of basis spectra integrated through each filter."""
    M = np.zeros((len(filters), basis_spectra.shape[0]))
    for i, filt in enumerate(filters):
        for j, basis in enumerate(basis_spectra):
            M[i, j] = compute_flux_in_band(basis, filt, wavelength)
    return M

def reconstruct_coeffs(photometry, filter_matrix):
    """Least‑squares solution for coefficients."""
    reg = LinearRegression(fit_intercept=False, normalize=True)
    reg.fit(filter_matrix.T, photometry)
    return reg.coef_

def reconstruct_spectrum(coeffs, basis_spectra):
    """Reconstruct full spectrum from coefficients."""
    return coeffs @ basis_spectra

# ----------------------------------------------------------------------
# Main demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.linspace(4000, 8000, 1000)  # Angstrom

    # Basis spectra
    n_basis = 5
    basis = create_basis_spectra(n_basis, wl)

    # True coefficients
    rng = np.random.default_rng(42)
    true_coeffs = rng.uniform(0.5, 1.5, size=n_basis)

    # True spectrum
    true_spec = generate_true_spectrum(true_coeffs, basis)

    # Filters (gaussian bandpasses)
    filter_centers = [4500, 5200, 5900, 6600, 7300]
    filter_widths  = [200, 200, 200, 200, 200]
    filters = [create_filter_response(c, w, wl) for c, w in zip(filter_centers, filter_widths)]

    # Photometric observations
    photometry = generate_photometry(true_spec, filters, wl)

    # Build filter matrix
    M = build_filter_matrix(filters, basis, wl)

    # Reconstruct coefficients
    rec_coeffs = reconstruct_coeffs(photometry, M)

    # Reconstructed spectrum
    rec_spec = reconstruct_spectrum(rec_coeffs, basis)

    # Print results
    print("True coefficients :", true_coeffs)
    print("Recovered coeffs:", rec_coeffs)
    print("\nPhotometry (true):", photometry)
    print("Photometry (recon.):", generate_photometry(rec_spec, filters, wl))
    print("\nSpectral error (RMS):", np.sqrt(np.mean((true_spec - rec_spec)**2)))