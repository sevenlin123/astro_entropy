import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------
# Spectral model – basis functions
# --------------------------------------------------------------------
def basis_functions(wl, n_basis=5):
    """Return a matrix of basis functions evaluated at wavelengths."""
    X = np.vstack([np.ones_like(wl), wl, wl**2, wl**3, wl**4])[:, :n_basis]
    return X

# --------------------------------------------------------------------
# Synthetic spectrum generator
# --------------------------------------------------------------------
def generate_synthetic_spectrum(wl, coeffs=None):
    """Generate a synthetic spectrum from random or provided coefficients."""
    if coeffs is None:
        coeffs = np.random.randn(5)  # 5‑dimensional basis
    X = basis_functions(wl, n_basis=len(coeffs))
    return X @ coeffs

# --------------------------------------------------------------------
# Filter definitions
# --------------------------------------------------------------------
def gaussian_filter(wl, center, width):
    """Gaussian filter transmission curve."""
    return np.exp(-0.5 * ((wl - center) / width)**2)

def create_filters(wl):
    """Create a small set of synthetic photometric filters."""
    centers = [450, 550, 650, 750]   # nm
    widths  = [20, 20, 20, 20]       # nm
    filters = [gaussian_filter(wl, c, w) for c, w in zip(centers, widths)]
    return np.array(filters)  # shape (n_filters, n_wl)

# --------------------------------------------------------------------
# Compute synthetic photometry
# --------------------------------------------------------------------
def compute_photometry(spectrum, filters):
    """Compute band‑integrated photometric fluxes from a spectrum."""
    fluxes = []
    for filt in filters:
        num = simps(spectrum * filt, x=wl)
        den = simps(filt, x=wl)
        fluxes.append(num / den)
    return np.array(fluxes)

# --------------------------------------------------------------------
# Reconstruction framework
# --------------------------------------------------------------------
def reconstruct_spectrum_from_photometry(photon_fluxes, filters, wl):
    """
    Reconstruct the spectrum coefficients that best reproduce the observed
    photometric fluxes using a linear regression on the filter responses.
    """
    # Build design matrix: each filter integrated against each basis
    X_design = []
    for filt in filters:
        # Integrate each basis function with the filter
        integrals = []
        for i in range(5):  # 5 basis functions
            basis = basis_functions(wl, n_basis=5)[:, i]
            integrals.append(simps(basis * filt, x=wl) / simps(filt, x=wl))
        X_design.append(integrals)
    X_design = np.array(X_design)  # shape (n_filters, n_basis)

    # Fit linear model: coeffs = (X^T X)^-1 X^T y
    reg = LinearRegression(fit_intercept=False).fit(X_design, photon_fluxes)
    coeffs = reg.coef_
    reconstructed = basis_functions(wl, n_basis=5) @ coeffs
    return coeffs, reconstructed

# --------------------------------------------------------------------
# Main demo
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid (400–800 nm)
    wl = np.linspace(400, 800, 200)

    # Generate a synthetic spectrum
    true_coeffs = np.array([1.0, -0.005, 0.00001, -2e-7, 1e-9])
    spectrum = generate_synthetic_spectrum(wl, true_coeffs)

    # Create filters and compute photometry
    filters = create_filters(wl)
    photometry = compute_photometry(spectrum, filters)

    # Reconstruct spectrum from photometry
    rec_coeffs, recon_spectrum = reconstruct_spectrum_from_photometry(
        photometry, filters, wl
    )

    # Print results
    print("True coefficients :", true_coeffs)
    print("Reconstructed coeffs:", rec_coeffs)
    print("\nDifference in spectra RMS:", np.sqrt(np.mean((spectrum - recon_spectrum)**2)))